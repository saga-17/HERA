import { useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import PipelineStatusPanel from "../components/PipelineStatusPanel";
import { askQuestion, uploadImage } from "../services/api";
import { useResultPolling } from "../hooks/useResultPolling";

export default function InferencePage() {
  const [imageId, setImageId] = useState<string | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [question, setQuestion] = useState("");
  const [uploading, setUploading] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [resultId, setResultId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const navigate = useNavigate();

  const { result, loading } = useResultPolling(resultId);

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setPreviewUrl(URL.createObjectURL(file));
    setUploading(true);
    setError(null);

    try {
      const response = await uploadImage(file);
      setImageId(response.image_id);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Upload failed");
    } finally {
      setUploading(false);
    }
  };

  const handleSubmit = async () => {
    if (!imageId || !question.trim()) return;

    setSubmitting(true);
    setError(null);

    try {
      const response = await askQuestion(imageId, question);
      setResultId(response.result_id);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Inference failed");
    } finally {
      setSubmitting(false);
    }
  };

  const isComplete = result?.pipeline_status.stage === "complete";

  if (isComplete && resultId) {
    navigate(`/result/${resultId}`);
  }

  return (
    <div className="max-w-7xl mx-auto px-4 py-8">
      <h1 className="text-3xl font-bold mb-8">Inference</h1>

      <div className="grid lg:grid-cols-3 gap-6">
        {/* Left: Image Upload */}
        <div className="card">
          <h3 className="text-lg font-semibold mb-4">Image Upload</h3>
          <div
            className="border-2 border-dashed border-slate-600 rounded-xl p-4 text-center cursor-pointer hover:border-hera-primary/50 transition-colors min-h-[280px] flex flex-col items-center justify-center"
            onClick={() => fileInputRef.current?.click()}
          >
            {previewUrl ? (
              <img
                src={previewUrl}
                alt="Preview"
                className="max-h-64 rounded-lg object-contain"
              />
            ) : (
              <>
                <div className="text-4xl mb-3">📷</div>
                <p className="text-slate-400">Click to upload an image</p>
                <p className="text-xs text-slate-500 mt-1">PNG, JPG, WEBP</p>
              </>
            )}
          </div>
          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            className="hidden"
            onChange={handleFileChange}
          />
          {uploading && <p className="text-sm text-indigo-400 mt-2">Uploading...</p>}
          {imageId && (
            <p className="text-xs text-green-400 mt-2">✓ Image uploaded</p>
          )}
        </div>

        {/* Middle: Question Input */}
        <div className="card flex flex-col">
          <h3 className="text-lg font-semibold mb-4">Question</h3>
          <textarea
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder="What do you see in this image? Ask a detailed question..."
            className="flex-1 bg-slate-800/50 border border-slate-600 rounded-xl p-4 text-white placeholder-slate-500 resize-none focus:outline-none focus:border-hera-primary min-h-[200px]"
          />
          <button
            onClick={handleSubmit}
            disabled={!imageId || !question.trim() || submitting || loading}
            className="btn-primary w-full mt-4"
          >
            {submitting || loading ? "Processing..." : "Run HERA Pipeline"}
          </button>
          {error && <p className="text-red-400 text-sm mt-2">{error}</p>}
        </div>

        {/* Right: Pipeline Status */}
        <PipelineStatusPanel
          status={result?.pipeline_status || null}
          loading={loading || submitting}
        />
      </div>
    </div>
  );
}
