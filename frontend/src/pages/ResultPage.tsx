import { useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";
import ReasoningStepCard from "../components/ReasoningStepCard";
import { getImageUrl, getResult } from "../services/api";
import type { HeraResult } from "../services/types";

const STRUCTURED_SECTION_LABELS: Record<string, string> = {
  observations: "Observations",
  reasoning: "Reasoning",
  conclusion: "Conclusion",
  answer: "Answer",
  corrected_reasoning: "Corrected Reasoning",
  final_reasoning: "Corrected Reasoning",
};

function stripKnownStructuralTags(text: string): string {
  return text.replace(
    /<\s*(?:\/\s*)?(?:observations|reasoning|conclusion|answer|corrected_reasoning|final_reasoning|corrected\s*[_ ]?reasoning|final\s*[_ ]?reasoning)\s*>/gi,
    ""
  ).trim();
}

function parseStructuredSections(rawText: string): { title: string; body: string }[] {
  const sections: { title: string; body: string }[] = [];
  const tagPattern = /<\s*([A-Za-z][A-Za-z0-9_ ]*?)\s*>([\s\S]*?)(?:<\/\s*\1\s*>|$)/gi;
  const matches = Array.from(rawText.matchAll(tagPattern));

  for (const match of matches) {
    const [, rawTag, content] = match;
    const normalizedTag = rawTag.replace(/[\s_]+/g, " ").trim().toLowerCase();
    const title = STRUCTURED_SECTION_LABELS[normalizedTag] ?? rawTag.trim();
    const body = content.trim();

    if (body && title && normalizedTag in STRUCTURED_SECTION_LABELS) {
      sections.push({ title, body });
    }
  }

  if (sections.length === 0) {
    const fallback = stripKnownStructuralTags(rawText);
    if (fallback) {
      sections.push({ title: "Reasoning", body: fallback });
    }
  }

  return sections;
}

function renderStructuredText(rawText: string) {
  const sections = parseStructuredSections(rawText);

  if (sections.length === 0) {
    return <pre className="text-sm text-slate-300 whitespace-pre-wrap font-sans leading-relaxed max-h-64 overflow-y-auto">{rawText || "No reasoning available."}</pre>;
  }

  return (
    <div className="space-y-4 max-h-64 overflow-y-auto pr-1">
      {sections.map((section) => (
        <div key={`${section.title}-${section.body.slice(0, 16)}`} className="rounded-lg border border-slate-700 bg-slate-900/40 p-3">
          <h4 className="text-xs uppercase tracking-wide text-slate-400 mb-2">{section.title}</h4>
          <div className="text-sm text-slate-200 whitespace-pre-wrap leading-relaxed">
            {section.body}
          </div>
        </div>
      ))}
    </div>
  );
}

function formatDuration(seconds: number | null | undefined): string {
  if (seconds === null || seconds === undefined || !Number.isFinite(seconds)) {
    return "--:--";
  }

  const totalSeconds = Math.max(0, Math.floor(seconds));
  const hours = Math.floor(totalSeconds / 3600);
  const minutes = Math.floor((totalSeconds % 3600) / 60);
  const secs = totalSeconds % 60;

  if (hours > 0) {
    return `${String(hours).padStart(2, "0")}:${String(minutes).padStart(2, "0")}:${String(secs).padStart(2, "0")}`;
  }

  return `${String(minutes).padStart(2, "0")}:${String(secs).padStart(2, "0")}`;
}

function normalizeText(value: string): string {
  return (value ?? "").replace(/\s+/g, " ").trim();
}

function toSafeHtml(value: string): string {
  return (value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/\"/g, "&quot;")
    .replace(/'/g, "&apos;");
}

function getVerificationStatus(result: HeraResult): "SUPPORTED" | "UNSUPPORTED" | "INCONCLUSIVE" {
  if (!result.steps.length) {
    return "INCONCLUSIVE";
  }

  const supported = result.steps.filter((step) => step.supported).length;
  const unsupported = result.steps.filter((step) => !step.supported).length;

  if (unsupported === 0 && supported > 0) {
    return "SUPPORTED";
  }

  if (supported === 0 && unsupported > 0) {
    return "UNSUPPORTED";
  }

  return "INCONCLUSIVE";
}

function getObservationsFromResult(result: HeraResult): string {
  const originalSections = parseStructuredSections(result.original_cot);
  const observations = originalSections.find((section) => section.title.toLowerCase() === "observations");
  if (observations?.body) {
    return observations.body;
  }

  const reasoning = result.original_cot || "";
  return normalizeText(reasoning) || "No observations were extracted.";
}

function buildReportText(result: HeraResult): string {
  const verificationStatus = getVerificationStatus(result);
  const observations = getObservationsFromResult(result);
  const evidenceLines = result.steps.flatMap((step) => {
    const lines: string[] = [];
    if (step.evidence) lines.push(`Step ${step.step_index + 1}: ${step.evidence}`);
    if (step.attribution) lines.push(`Attribution: ${step.attribution}`);
    if (step.textual_evidence.length > 0) {
      lines.push(
        ...step.textual_evidence.map((e) => `Text evidence: ${e.text}`)
      );
    }
    if (step.visual_evidence.length > 0) {
      lines.push(
        ...step.visual_evidence.map((e) => `Visual evidence: ${e.caption || e.region_id}`)
      );
    }
    return lines;
  });

  const sections = [
    "HERA-VLM ANALYSIS REPORT",
    "",
    "Input Image",
    "",
    `Image ID: ${result.image_id}`,
    `Question: ${result.question}`,
    "",
    "Final Grounded Answer",
    "",
    result.final_answer || "No final answer available.",
    "",
    "Verification Status",
    "",
    verificationStatus,
    "",
    "Confidence Score",
    "",
    `${(result.confidence_score * 100).toFixed(0)}%`,
    "",
    "Hallucination Score",
    "",
    `${(result.hallucination_score * 100).toFixed(0)}%`,
    "",
    "Pipeline Summary",
    "",
    `Total stages: ${result.pipeline_status.total_stages || 0}`,
    `Completed stages: ${result.pipeline_status.completed_stages || 0}`,
    `Execution time: ${formatDuration(result.pipeline_status.elapsed_seconds)}`,
    "",
    "Observations",
    "",
    observations,
    "",
    "Evidence",
    "",
    evidenceLines.length ? evidenceLines.join("\n") : "No evidence captured.",
    "",
    "Reasoning / Analysis",
    "",
    result.original_cot || "No original reasoning available.",
    "",
    "Corrected Reasoning",
    "",
    result.corrected_cot || "No corrected reasoning available.",
    "",
    "Evidence Attribution",
    "",
    result.attributed_cot || "No evidence attribution available.",
  ];

  return sections.join("\n");
}

function buildHtmlReport(result: HeraResult): string {
  const imageUrl = getImageUrl(result.image_id);
  const verificationStatus = getVerificationStatus(result);
  const observations = getObservationsFromResult(result);
  const evidenceHtml = result.steps.length
    ? result.steps
        .map(
          (step) => `
            <div class="section-block">
              <h3>Step ${step.step_index + 1}</h3>
              <p><strong>Status:</strong> ${step.status}</p>
              <p><strong>Claim:</strong> ${toSafeHtml(step.step)}</p>
              ${step.evidence ? `<p><strong>Evidence:</strong> ${toSafeHtml(step.evidence)}</p>` : ""}
              ${step.attribution ? `<p><strong>Attribution:</strong> ${toSafeHtml(step.attribution)}</p>` : ""}
              ${step.textual_evidence.length ? `<ul>${step.textual_evidence.map((e) => `<li>${toSafeHtml(e.text)}</li>`).join("")}</ul>` : ""}
            </div>
          `
        )
        .join("")
    : "<p>No evidence captured.</p>";

  return `<!doctype html>
    <html lang="en">
      <head>
        <meta charset="UTF-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1.0" />
        <title>HERA-VLM Analysis Report</title>
        <style>
          @page { size: A4; margin: 12mm; }
          body {
            margin: 0;
            padding: 24px;
            font-family: Arial, Helvetica, sans-serif;
            background: white;
            color: #111827;
            line-height: 1.6;
          }
          .report {
            max-width: 820px;
            margin: 0 auto;
          }
          h1, h2, h3 { color: #111827; }
          .header {
            border-bottom: 2px solid #e5e7eb;
            padding-bottom: 16px;
            margin-bottom: 20px;
          }
          .meta-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(180px, 1fr));
            gap: 12px;
            margin: 20px 0;
          }
          .card {
            border: 1px solid #e5e7eb;
            border-radius: 10px;
            padding: 12px 14px;
            background: #f9fafb;
          }
          .image-box {
            margin: 16px 0;
            text-align: center;
          }
          img {
            max-width: 100%;
            max-height: 440px;
            border: 1px solid #d1d5db;
            border-radius: 8px;
          }
          .section-block {
            padding: 14px 0;
            border-top: 1px solid #e5e7eb;
            margin-top: 12px;
          }
          ul { padding-left: 18px; }
          @media print {
            body { padding: 0; }
            .report { max-width: none; }
          }
        </style>
      </head>
      <body>
        <main class="report">
          <div class="header">
            <h1>HERA-VLM ANALYSIS REPORT</h1>
          </div>

          <div class="image-box">
            <img src="${imageUrl}" alt="Input image" />
          </div>

          <div class="meta-grid">
            <div class="card"><strong>User Query</strong><br/>${toSafeHtml(result.question)}</div>
            <div class="card"><strong>Verification Status</strong><br/>${verificationStatus}</div>
            <div class="card"><strong>Confidence Score</strong><br/>${(result.confidence_score * 100).toFixed(0)}%</div>
            <div class="card"><strong>Hallucination Score</strong><br/>${(result.hallucination_score * 100).toFixed(0)}%</div>
            <div class="card"><strong>Total Stages</strong><br/>${result.pipeline_status.total_stages || 0}</div>
            <div class="card"><strong>Completed Stages</strong><br/>${result.pipeline_status.completed_stages || 0}</div>
            <div class="card"><strong>Execution Time</strong><br/>${formatDuration(result.pipeline_status.elapsed_seconds)}</div>
            <div class="card"><strong>Final Answer</strong><br/>${toSafeHtml(result.final_answer || "No final answer available.")}</div>
          </div>

          <div class="section-block">
            <h2>Observations</h2>
            <p>${toSafeHtml(observations)}</p>
          </div>

          <div class="section-block">
            <h2>Evidence</h2>
            ${evidenceHtml}
          </div>

          <div class="section-block">
            <h2>Reasoning / Analysis</h2>
            <p>${toSafeHtml(result.original_cot || "No original reasoning available.")}</p>
          </div>

          <div class="section-block">
            <h2>Corrected Reasoning</h2>
            <p>${toSafeHtml(result.corrected_cot || "No corrected reasoning available.")}</p>
          </div>

          <div class="section-block">
            <h2>Evidence Attribution</h2>
            <p>${toSafeHtml(result.attributed_cot || "No evidence attribution available.")}</p>
          </div>
        </main>
      </body>
    </html>`;
}

function createBlobFromText(content: string, mimeType: string): Blob {
  return new Blob([content], { type: mimeType });
}

function downloadBlob(filename: string, blob: Blob): void {
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  anchor.click();
  URL.revokeObjectURL(url);
}

function openPrintWindow(reportHtml: string): void {
  const printWindow = window.open("", "_blank", "width=1200,height=900");
  if (!printWindow) {
    return;
  }

  printWindow.document.write(reportHtml);
  printWindow.document.close();
  printWindow.focus();
  printWindow.print();
}

function crc32(data: Uint8Array): number {
  let crc = 0xffffffff;
  for (let i = 0; i < data.length; i += 1) {
    crc ^= data[i];
    for (let bit = 0; bit < 8; bit += 1) {
      const isSet = (crc & 1) !== 0;
      crc = (crc >>> 1) ^ (isSet ? 0xedb88320 : 0);
    }
  }
  return (crc ^ 0xffffffff) >>> 0;
}

function buildDocxFromText(text: string): Blob {
  const documentXml = (() => {
    const paragraphs = text
      .split(/\n{2,}/)
      .map((block) => block.trim())
      .filter(Boolean)
      .map((block) => {
        const isHeading = block.includes(":") && block.length < 120;
        const escaped = toSafeHtml(block).replace(/\n/g, "<w:br/>");
        return isHeading
          ? `<w:p><w:pPr><w:pStyle w:val="Heading1"/></w:pPr><w:r><w:t xml:space="preserve">${escaped}</w:t></w:r></w:p>`
          : `<w:p><w:r><w:t xml:space="preserve">${escaped}</w:t></w:r></w:p>`;
      })
      .join("");

    return `<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
      <w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
        <w:body>
          ${paragraphs}
          <w:sectPr>
            <w:pgSz w:w="12240" w:h="15840"/>
            <w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440" w:header="720" w:footer="720" w:gutter="0"/>
          </w:sectPr>
        </w:body>
      </w:document>`;
  })();

  const files: Array<{ name: string; content: Uint8Array }> = [
    {
      name: "[Content_Types].xml",
      content: new TextEncoder().encode(`<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
        <Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
          <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
          <Default Extension="xml" ContentType="application/xml"/>
          <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
          <Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
          <Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>
        </Types>`),
    },
    {
      name: "_rels/.rels",
      content: new TextEncoder().encode(`<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
        <Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
          <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
          <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
          <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>
        </Relationships>`),
    },
    {
      name: "docProps/core.xml",
      content: new TextEncoder().encode(`<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
        <cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:dcmitype="http://purl.org/dc/dcmitype/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
          <dc:title>HERA-VLM Analysis Report</dc:title>
          <dc:creator>HERA-VLM</dc:creator>
          <cp:lastModifiedBy>HERA-VLM</cp:lastModifiedBy>
          <dcterms:created xsi:type="dcterms:W3CDTF">2026-09-10T00:00:00Z</dcterms:created>
          <dcterms:modified xsi:type="dcterms:W3CDTF">2026-09-10T00:00:00Z</dcterms:modified>
        </cp:coreProperties>`),
    },
    {
      name: "docProps/app.xml",
      content: new TextEncoder().encode(`<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
        <Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">
          <Application>HERA-VLM</Application>
        </Properties>`),
    },
    {
      name: "word/document.xml",
      content: new TextEncoder().encode(documentXml),
    },
  ];

  const localHeaders: Uint8Array[] = [];
  const centralDirectory: Uint8Array[] = [];
  let offset = 0;

  for (const file of files) {
    const nameBytes = new TextEncoder().encode(file.name);
    const header = new Uint8Array(30 + nameBytes.length);
    const view = new DataView(header.buffer);
    view.setUint32(0, 0x04034b50, true);
    view.setUint16(4, 20, true);
    view.setUint16(6, 0, true);
    view.setUint16(8, 0, true);
    view.setUint16(10, 0, true);
    view.setUint16(12, 0, true);
    view.setUint32(14, crc32(file.content), true);
    view.setUint32(18, file.content.length, true);
    view.setUint32(22, file.content.length, true);
    view.setUint16(26, nameBytes.length, true);
    view.setUint16(28, 0, true);

    header.set(nameBytes, 30);
    localHeaders.push(header, file.content);

    const centralHeader = new Uint8Array(46 + nameBytes.length);
    const centralView = new DataView(centralHeader.buffer);
    centralView.setUint32(0, 0x02014b50, true);
    centralView.setUint16(4, 20, true);
    centralView.setUint16(6, 20, true);
    centralView.setUint16(8, 0, true);
    centralView.setUint16(10, 0, true);
    centralView.setUint16(12, 0, true);
    centralView.setUint16(14, 0, true);
    centralView.setUint32(16, crc32(file.content), true);
    centralView.setUint32(20, file.content.length, true);
    centralView.setUint32(24, file.content.length, true);
    centralView.setUint16(28, nameBytes.length, true);
    centralView.setUint16(30, 0, true);
    centralView.setUint16(32, 0, true);
    centralView.setUint16(34, 0, true);
    centralView.setUint16(36, 0, true);
    centralView.setUint32(38, 0, true);
    centralView.setUint32(42, offset, true);
    centralHeader.set(nameBytes, 46);
    centralDirectory.push(centralHeader);

    offset += header.length + file.content.length;
  }

  const centralSize = centralDirectory.reduce((total, entry) => total + entry.length, 0);
  const eocd = new Uint8Array(22);
  const eocdView = new DataView(eocd.buffer);
  eocdView.setUint32(0, 0x06054b50, true);
  eocdView.setUint16(4, 0, true);
  eocdView.setUint16(6, 0, true);
  eocdView.setUint16(8, files.length, true);
  eocdView.setUint16(10, files.length, true);
  eocdView.setUint32(12, centralSize, true);
  eocdView.setUint32(16, offset, true);
  eocdView.setUint16(20, 0, true);

  const archive = new Uint8Array(
    localHeaders.reduce((total, entry) => total + entry.length, 0) + centralSize + eocd.length
  );
  let cursor = 0;

  for (const chunk of localHeaders) {
    archive.set(chunk, cursor);
    cursor += chunk.length;
  }

  for (const entry of centralDirectory) {
    archive.set(entry, cursor);
    cursor += entry.length;
  }

  archive.set(eocd, cursor);

  return new Blob([archive], { type: "application/vnd.openxmlformats-officedocument.wordprocessingml.document" });
}

function exportReport(format: "json" | "txt" | "html" | "pdf" | "docx", result: HeraResult): void {
  if (format === "json") {
    downloadBlob("hera-vlm-report.json", createBlobFromText(JSON.stringify(result, null, 2), "application/json"));
    return;
  }

  if (format === "txt") {
    downloadBlob("hera-vlm-report.txt", createBlobFromText(buildReportText(result), "text/plain;charset=utf-8"));
    return;
  }

  if (format === "html") {
    downloadBlob("hera-vlm-report.html", createBlobFromText(buildHtmlReport(result), "text/html;charset=utf-8"));
    return;
  }

  if (format === "docx") {
    downloadBlob("hera-vlm-report.docx", buildDocxFromText(buildReportText(result)));
    return;
  }

  if (format === "pdf") {
    openPrintWindow(buildHtmlReport(result));
  }
}

export default function ResultPage() {
  const { resultId } = useParams<{ resultId: string }>();
  const [result, setResult] = useState<HeraResult | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!resultId) return;
    getResult(resultId)
      .then(setResult)
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false));
  }, [resultId]);

  if (loading) {
    return (
      <div className="max-w-7xl mx-auto px-4 py-16 text-center">
        <div className="animate-pulse text-slate-400">Loading results...</div>
      </div>
    );
  }

  if (error || !result) {
    return (
      <div className="max-w-7xl mx-auto px-4 py-16 text-center">
        <p className="text-red-400 mb-4">{error || "Result not found"}</p>
        <Link to="/inference" className="btn-primary">Try Again</Link>
      </div>
    );
  }

  const hallucinationPct = (result.hallucination_score * 100).toFixed(0);
  const confidencePct = (result.confidence_score * 100).toFixed(0);
  const verificationStatus = getVerificationStatus(result);

  return (
    <div className="max-w-7xl mx-auto px-4 py-8">
      <div className="flex items-center justify-between mb-8">
        <h1 className="text-3xl font-bold">Results</h1>
        <Link to="/inference" className="btn-secondary">New Inference</Link>
      </div>

      <div className="grid md:grid-cols-4 gap-4 mb-8">
        <div className="card text-center">
          <div className="text-xs text-slate-400 mb-1">Hallucination Score</div>
          <div className={`text-3xl font-bold ${result.hallucination_score > 0.3 ? "text-red-400" : "text-green-400"}`}>
            {hallucinationPct}%
          </div>
        </div>
        <div className="card text-center">
          <div className="text-xs text-slate-400 mb-1">Confidence Score</div>
          <div className="text-3xl font-bold text-indigo-400">{confidencePct}%</div>
        </div>
        <div className="card text-center">
          <div className="text-xs text-slate-400 mb-1">Total Steps</div>
          <div className="text-3xl font-bold">{result.steps.length}</div>
        </div>
        <div className="card text-center">
          <div className="text-xs text-slate-400 mb-1">Hallucinated Steps</div>
          <div className="text-3xl font-bold text-red-400">
            {result.steps.filter((s) => !s.supported).length}
          </div>
        </div>
      </div>

      <div className="card mb-8">
        <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4 mb-4">
          <div>
            <h2 className="text-2xl font-bold mb-1">Export / Report</h2>
            <p className="text-slate-400 text-sm">Generate a professional report for sharing, printing, or archive.</p>
          </div>
          <button
            type="button"
            className="btn-primary"
            onClick={() => openPrintWindow(buildHtmlReport(result))}
          >
            Print Report
          </button>
        </div>

        <div className="flex flex-wrap gap-3">
          <button type="button" className="btn-secondary" onClick={() => exportReport("pdf", result)}>
            PDF
          </button>
          <button type="button" className="btn-secondary" onClick={() => exportReport("docx", result)}>
            DOCX
          </button>
          <button type="button" className="btn-secondary" onClick={() => exportReport("html", result)}>
            HTML
          </button>
          <button type="button" className="btn-secondary" onClick={() => exportReport("txt", result)}>
            TXT
          </button>
          <button type="button" className="btn-secondary" onClick={() => exportReport("json", result)}>
            JSON
          </button>
        </div>
      </div>

      <div className="grid lg:grid-cols-2 gap-6 mb-8">
        <div className="card">
          <h3 className="font-semibold mb-3">Input</h3>
          <img
            src={getImageUrl(result.image_id)}
            alt="Uploaded"
            className="w-full max-h-64 object-contain rounded-lg mb-4"
          />
          <p className="text-slate-300">
            <span className="text-slate-500">Q:</span> {result.question}
          </p>
        </div>

        <div className="card">
          <h3 className="font-semibold mb-3">Final Grounded Answer</h3>
          <p className="text-lg text-green-300 leading-relaxed">{result.final_answer}</p>
          <div className="mt-6 grid grid-cols-2 gap-3 text-sm">
            <div className="rounded-lg border border-slate-700 bg-slate-900/40 p-3">
              <div className="text-slate-400">Verification</div>
              <div className="font-semibold mt-1">{verificationStatus}</div>
            </div>
            <div className="rounded-lg border border-slate-700 bg-slate-900/40 p-3">
              <div className="text-slate-400">Pipeline Time</div>
              <div className="font-semibold mt-1">{formatDuration(result.pipeline_status.elapsed_seconds)}</div>
            </div>
          </div>
        </div>
      </div>

      <div className="grid lg:grid-cols-2 gap-6 mb-8">
        <div className="card">
          <h3 className="font-semibold mb-3">Original Chain-of-Thought</h3>
          {renderStructuredText(result.original_cot)}
        </div>
        <div className="card">
          <h3 className="font-semibold mb-3">Evidence-Attributed CoT</h3>
          {renderStructuredText(result.attributed_cot)}
        </div>
      </div>

      {result.corrected_cot && (
        <div className="card mb-8">
          <h3 className="font-semibold mb-3">Corrected Reasoning</h3>
          {renderStructuredText(result.corrected_cot)}
        </div>
      )}

      <h2 className="text-2xl font-bold mb-4">Reasoning Steps</h2>
      <div className="space-y-4">
        {result.steps.map((step) => (
          <ReasoningStepCard key={step.step_index} step={step} />
        ))}
      </div>
    </div>
  );
}
