const SPACE_ID = "shaanzeeeee/vit-base-pc-parts-inference";
const API_NAME = "/predict";

const CLASS_HELP = {
  AIO_Liquid_Cooler: "All-in-one liquid cooling systems for CPUs.",
  Air_Cooler: "Traditional fan + heatsink CPU coolers.",
  Bad_Cable_Management: "PC build with cluttered or obstructive cable routing.",
  CPU: "Central Processing Unit related visuals or packaging.",
  Good_Cable_Management: "PC build with clean and organized cable routing.",
  Graphics_Card: "Discrete GPU cards and related board visuals.",
  M2_NVMe_Drive: "M.2 NVMe storage devices.",
  Motherboard: "Mainboard components and full motherboard shots.",
  PC_Case: "Computer chassis and enclosure images.",
  Power_Supply: "PSU units and related component visuals.",
  RAM_Stick: "Memory modules (DIMMs).",
};

const dropzone = document.getElementById("dropzone");
const imageInput = document.getElementById("imageInput");
const predictBtn = document.getElementById("predictBtn");
const clearBtn = document.getElementById("clearBtn");
const copyBtn = document.getElementById("copyBtn");
const downloadBtn = document.getElementById("downloadBtn");

const statusEl = document.getElementById("status");
const previewWrap = document.getElementById("previewWrap");
const previewImage = document.getElementById("previewImage");

const resultsCard = document.getElementById("resultsCard");
const resultEmpty = document.getElementById("resultEmpty");
const resultContent = document.getElementById("resultContent");
const topClass = document.getElementById("topClass");
const topConfidence = document.getElementById("topConfidence");
const topList = document.getElementById("topList");

const classHelp = document.getElementById("classHelp");

let selectedFile = null;
let lastResult = null;
let appClient = null;

function renderClassHelp() {
  const items = Object.entries(CLASS_HELP)
    .map(([name, desc]) => `<li><strong>${name}</strong>: ${desc}</li>`)
    .join("");
  classHelp.innerHTML = items;
}

function setStatus(message, isError = false) {
  statusEl.textContent = message;
  statusEl.classList.toggle("error", isError);
}

function setBusy(isBusy) {
  predictBtn.disabled = isBusy || !selectedFile;
  clearBtn.disabled = isBusy && !selectedFile;
  copyBtn.disabled = isBusy || !lastResult;
  downloadBtn.disabled = isBusy || !lastResult;
  predictBtn.textContent = isBusy ? "Analyzing..." : "Classify Image";
}

function validateFile(file) {
  if (!file) return "No file selected.";
  if (!file.type.startsWith("image/")) return "Please upload a valid image file.";
  const maxSizeMb = 12;
  if (file.size > maxSizeMb * 1024 * 1024) {
    return `Image is too large. Max allowed size is ${maxSizeMb} MB.`;
  }
  return null;
}

function showPreview(file) {
  const url = URL.createObjectURL(file);
  previewImage.src = url;
  previewWrap.classList.remove("hidden");
}

function clearPreview() {
  previewImage.removeAttribute("src");
  previewWrap.classList.add("hidden");
}

function normalizePredictions(raw) {
  if (!raw) return [];

  if (raw.confidences && Array.isArray(raw.confidences)) {
    return raw.confidences
      .map((c) => ({ label: String(c.label), confidence: Number(c.confidence) }))
      .sort((a, b) => b.confidence - a.confidence);
  }

  if (typeof raw === "object" && !Array.isArray(raw)) {
    return Object.entries(raw)
      .map(([label, confidence]) => ({ label, confidence: Number(confidence) }))
      .filter((x) => Number.isFinite(x.confidence))
      .sort((a, b) => b.confidence - a.confidence);
  }

  return [];
}

function confidenceTag(score) {
  if (score >= 0.8) return "High";
  if (score >= 0.55) return "Medium";
  return "Low";
}

function renderResults(list) {
  if (!list.length) {
    resultContent.classList.add("hidden");
    resultEmpty.classList.remove("hidden");
    setStatus("No predictions returned from endpoint.", true);
    return;
  }

  const top = list[0];
  topClass.textContent = top.label;
  topConfidence.textContent = `${(top.confidence * 100).toFixed(2)}% (${confidenceTag(top.confidence)} confidence)`;

  const top5 = list.slice(0, 5);
  topList.innerHTML = top5
    .map((item) => {
      const pct = Math.max(0, Math.min(100, item.confidence * 100));
      return `
        <li class="top-item">
          <div class="top-item-head">
            <span class="class-name">${item.label}</span>
            <span class="score">${pct.toFixed(2)}%</span>
          </div>
          <div class="bar"><span style="width:${pct.toFixed(2)}%"></span></div>
        </li>
      `;
    })
    .join("");

  resultEmpty.classList.add("hidden");
  resultContent.classList.remove("hidden");
  resultsCard.scrollIntoView({ behavior: "smooth", block: "start" });
}

async function getClient() {
  if (appClient) return appClient;

  const module = await import("https://cdn.jsdelivr.net/npm/@gradio/client/dist/index.min.js");
  const clientFactory = module.client || module.Client || module.default?.client;

  if (!clientFactory) {
    throw new Error("Could not load Gradio client library.");
  }

  appClient = await clientFactory(SPACE_ID);
  return appClient;
}

async function runPrediction() {
  if (!selectedFile) {
    setStatus("Please upload an image first.", true);
    return;
  }

  setBusy(true);
  setStatus("Analyzing image...");

  try {
    const app = await getClient();
    const response = await app.predict(API_NAME, { image: selectedFile });

    // Response can be {data:[...]} or direct payload depending on client version
    const payload = Array.isArray(response?.data) ? response.data[0] : response;
    const normalized = normalizePredictions(payload);

    lastResult = {
      fileName: selectedFile.name,
      timestamp: new Date().toISOString(),
      predictions: normalized,
    };

    renderResults(normalized);
    copyBtn.disabled = false;
    downloadBtn.disabled = false;
    setStatus("Prediction complete.");
  } catch (err) {
    console.error(err);
    const msg = err?.message || "Inference request failed.";
    setStatus(`Error: ${msg}`, true);
  } finally {
    setBusy(false);
  }
}

function resetAll() {
  selectedFile = null;
  lastResult = null;
  imageInput.value = "";
  clearPreview();

  resultContent.classList.add("hidden");
  resultEmpty.classList.remove("hidden");
  topList.innerHTML = "";
  topClass.textContent = "-";
  topConfidence.textContent = "-";

  copyBtn.disabled = true;
  downloadBtn.disabled = true;
  predictBtn.disabled = true;
  clearBtn.disabled = true;
  setStatus("", false);
}

async function copyResults() {
  if (!lastResult) return;
  const text = [
    `File: ${lastResult.fileName}`,
    `Timestamp: ${lastResult.timestamp}`,
    "Top-5 Predictions:",
    ...lastResult.predictions.slice(0, 5).map((p, i) => `${i + 1}. ${p.label}: ${(p.confidence * 100).toFixed(2)}%`),
  ].join("\n");

  try {
    await navigator.clipboard.writeText(text);
    setStatus("Results copied to clipboard.");
  } catch {
    setStatus("Could not copy to clipboard.", true);
  }
}

function downloadResults() {
  if (!lastResult) return;
  const blob = new Blob([JSON.stringify(lastResult, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "prediction-results.json";
  a.click();
  URL.revokeObjectURL(url);
  setStatus("Results downloaded.");
}

function handleFile(file) {
  const error = validateFile(file);
  if (error) {
    setStatus(error, true);
    return;
  }

  selectedFile = file;
  showPreview(file);
  setStatus("Image loaded. Click 'Classify Image' to run inference.");
  predictBtn.disabled = false;
  clearBtn.disabled = false;
}

imageInput.addEventListener("change", (e) => {
  const file = e.target.files?.[0];
  if (file) handleFile(file);
});

["dragenter", "dragover"].forEach((eventName) => {
  dropzone.addEventListener(eventName, (e) => {
    e.preventDefault();
    e.stopPropagation();
    dropzone.classList.add("dragover");
  });
});

["dragleave", "drop"].forEach((eventName) => {
  dropzone.addEventListener(eventName, (e) => {
    e.preventDefault();
    e.stopPropagation();
    dropzone.classList.remove("dragover");
  });
});

dropzone.addEventListener("drop", (e) => {
  const file = e.dataTransfer?.files?.[0];
  if (file) handleFile(file);
});

dropzone.addEventListener("keydown", (e) => {
  if (e.key === "Enter" || e.key === " ") {
    e.preventDefault();
    imageInput.click();
  }
});

predictBtn.addEventListener("click", runPrediction);
clearBtn.addEventListener("click", resetAll);
copyBtn.addEventListener("click", copyResults);
downloadBtn.addEventListener("click", downloadResults);

renderClassHelp();
resetAll();
