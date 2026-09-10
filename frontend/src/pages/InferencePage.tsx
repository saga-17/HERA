import { useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import PipelineStatusPanel from "../components/PipelineStatusPanel";
import { askQuestion, cancelInference, uploadImage } from "../services/api";
import { useResultPolling } from "../hooks/useResultPolling";

type PipelineStatusValue = "idle" | "running" | "completed" | "failed" | "cancelled";

export default function InferencePage() {
  const [imageId, setImageId] = useState<string | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [question, setQuestion] = useState("");
  const [uploading, setUploading] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [resultId, setResultId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [pipelineStatus, setPipelineStatus] = useState<PipelineStatusValue>("idle");
  const fileInputRef = useRef<HTMLInputElement>(null);
  const promptRef = useRef<HTMLTextAreaElement>(null);
  const navigate = useNavigate();

  const { result, loading } = useResultPolling(resultId);

  useEffect(() => {
    if (!result) {
      return;
    }

    const stage = result.pipeline_status.stage?.toLowerCase?.() ?? "";
    if (stage === "complete") {
      setPipelineStatus("completed");
      return;
    }
    if (stage === "error") {
      setPipelineStatus("failed");
      return;
    }
    if (stage === "cancelled") {
      setPipelineStatus("cancelled");
      return;
    }
    setPipelineStatus("running");
  }, [result]);

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

  const isRunning = pipelineStatus === "running" || submitting || loading;

  const handleSubmit = async () => {
    if (!imageId || !question.trim() || isRunning) return;

    setSubmitting(true);
    setError(null);
    setPipelineStatus("running");
    setResultId(null);

    try {
      const response = await askQuestion(imageId, question);
      setResultId(response.result_id);
      setPipelineStatus(response.status === "running" ? "running" : "running");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Inference failed");
      setPipelineStatus("failed");
    } finally {
      setSubmitting(false);
    }
  };

  const handleEditRestart = async () => {
    if (!resultId) {
      setPipelineStatus("idle");
      promptRef.current?.focus();
      return;
    }

    const confirmRestart = window.confirm(
      "The current pipeline will be stopped and restarted with the new prompt."
    );

    if (!confirmRestart) {
      return;
    }

    try {
      await cancelInference(resultId);
    } catch (err) {
      console.warn("Pipeline cancel skipped", err);
    }

    setResultId(null);
    setPipelineStatus("idle");
    setError(null);
    setSubmitting(false);
    promptRef.current?.focus();
  };

  const handleTerminateProcess = async () => {
    if (!resultId) {
      return;
    }

    const confirmTerminate = window.confirm(
      "Terminate the current HERA pipeline?\nThe current processing will be stopped."
    );

    if (!confirmTerminate) {
      return;
    }

    try {
      await cancelInference(resultId);
    } catch (err) {
      console.warn("Terminate failed", err);
    }

    setResultId(null);
    setPipelineStatus("cancelled");
    setSubmitting(false);
    setError(null);
  };

  const isComplete = result?.pipeline_status.stage === "complete";

  useEffect(() => {
    if (isComplete && resultId) {
      navigate(`/result/${resultId}`);
    }
  }, [isComplete, resultId, navigate]);

  return (
    <div className="max-w-7xl mx-auto px-4 py-8">
      <h1 className="text-3xl font-bold mb-8">Inference</h1>

      <div className="grid gap-6 xl:grid-cols-3 lg:grid-cols-2 md:grid-cols-1">
        {/* Left: Image Upload */}
        <div className="card min-w-0">
          <h3 className="text-lg font-semibold mb-4">Image Upload</h3>
          <div
            className="border-2 border-dashed border-slate-600 rounded-xl p-4 text-center cursor-pointer hover:border-hera-primary/50 transition-colors min-h-[280px] flex flex-col items-center justify-center"
            onClick={() => fileInputRef.current?.click()}
          >
            {previewUrl ? (
              <img
                src={previewUrl}
                alt="Preview"
                className="max-h-64 w-full rounded-lg object-contain"
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
        <div className="card flex flex-col min-w-0">
          <h3 className="text-lg font-semibold mb-4">Question</h3>

          {isRunning && (
            <div className="mb-3 rounded-lg border border-amber-500/30 bg-amber-500/10 px-3 py-2 text-sm text-amber-300">
              Pipeline running...
            </div>
          )}

          <textarea
            ref={promptRef}
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder="What do you see in this image? Ask a detailed question..."
            disabled={isRunning}
            className="flex-1 bg-slate-800/50 border border-slate-600 rounded-xl p-4 text-white placeholder-slate-500 resize-none focus:outline-none focus:border-hera-primary min-h-[200px] disabled:opacity-75 disabled:cursor-not-allowed"
          />

          <div className="mt-4 flex flex-col gap-2 sm:flex-row">
            {isRunning ? (
              <button
                type="button"
                className="btn-primary flex-1 disabled:cursor-not-allowed"
                disabled
              >
                Running...
              </button>
            ) : (
              <button
                type="button"
                onClick={handleSubmit}
                disabled={!imageId || !question.trim()}
                className="btn-primary flex-1"
              >
                Run HERA Pipeline
              </button>
            )}
          </div>

          {isRunning && (
            <div className="mt-3 flex flex-col gap-2 sm:flex-row">
              <button
                type="button"
                onClick={handleEditRestart}
                className="btn-secondary flex-1"
              >
                Edit Prompt & Restart
              </button>
              <button
                type="button"
                onClick={handleTerminateProcess}
                className="bg-red-600 hover:bg-red-500 text-white font-medium px-4 py-2 rounded-lg transition-colors"
              >
                Terminate Process
              </button>
            </div>
          )}

          {error && <p className="text-red-400 text-sm mt-2">{error}</p>}
        </div>

        {/* Right: Pipeline Status */}
        <PipelineStatusPanel
          status={result?.pipeline_status || null}
          loading={isRunning}
        />
      </div>
    </div>
  );
}
