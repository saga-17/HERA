import json
import time
import urllib.request
from pathlib import Path
from tempfile import gettempdir

from PIL import Image

BASE = "http://127.0.0.1:8000"


def http_json(url, payload=None, method="GET", headers=None):
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(url, data=data, headers=headers or {}, method=method)
    with urllib.request.urlopen(request, timeout=60) as response:
        body = response.read()
        return json.loads(body.decode()) if body else {}


def upload_image():
    image_path = Path(gettempdir()) / "hera_api_lifecycle.png"
    Image.new("RGB", (64, 64), color=(12, 34, 200)).save(image_path)
    boundary = "----HERA_API_TEST"
    body = b"".join([
        f"--{boundary}\r\n".encode(),
        b'Content-Disposition: form-data; name="file"; filename="hera_api_lifecycle.png"\r\n',
        b"Content-Type: image/png\r\n\r\n",
        image_path.read_bytes(),
        f"\r\n--{boundary}--\r\n".encode(),
    ])
    request = urllib.request.Request(
        f"{BASE}/api/upload-image",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode())


image = upload_image()
print("UPLOAD", image)

first_request = {
    "image_id": image["image_id"],
    "question": "Describe the scene.",
}
first = http_json(f"{BASE}/api/ask", first_request, method="POST", headers={"Content-Type": "application/json"})
second = http_json(f"{BASE}/api/ask", first_request, method="POST", headers={"Content-Type": "application/json"})
print("FIRST_ASK", first)
print("SECOND_ASK", second)
assert first["result_id"] == second["result_id"], "duplicate request did not reuse active run"

# Give the background task a moment to start and then cancel it.
time.sleep(2)
cancel = http_json(f"{BASE}/api/cancel/{first['result_id']}", method="POST")
print("CANCEL", cancel)

result = http_json(f"{BASE}/api/result/{first['result_id']}")
print("RESULT", result["pipeline_status"]["stage"], result["pipeline_status"]["message"])
assert result["pipeline_status"]["stage"] == "cancelled", result
