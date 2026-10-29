(function () {
  "use strict";

  const CLASS_ORDER = ["fear", "happy", "neutral", "sad"];
  const API_BASE_URL = (window.EEG_API_BASE_URL || "http://127.0.0.1:8000").replace(/\/$/, "");
  const elements = {
    healthIndicator: document.getElementById("health-indicator"),
    healthText: document.getElementById("health-text"),
    moduleStartPanel: document.getElementById("module-start-panel"),
    moduleApiStatus: document.getElementById("module-api-status"),
    moduleStartMessage: document.getElementById("module-start-message"),
    startModulesButton: document.getElementById("start-modules-button"),
    uploadPanel: document.getElementById("upload-panel"),
    flowSteps: Array.from(document.querySelectorAll("[data-flow-step]")),
    uploadZone: document.getElementById("upload-zone"),
    fileInput: document.getElementById("file-input"),
    fileName: document.getElementById("file-name"),
    error: document.getElementById("error-message"),
    predictButton: document.getElementById("predict-button"),
    predictButtonLabel: document.getElementById("predict-button-label"),
    resetButton: document.getElementById("reset-button"),
    resultsResetButton: document.getElementById("results-reset-button"),
    uploadStatus: document.querySelector("[data-upload-status]"),
    results: document.getElementById("results"),
    resultFile: document.getElementById("result-file"),
    resultRecordMeta: document.getElementById("result-record-meta"),
    resultEmotion: document.getElementById("result-emotion"),
    resultConfidence: document.getElementById("result-confidence"),
    confidenceDescription: document.getElementById("confidence-description"),
    probabilityList: document.getElementById("probability-list"),
    windowCounts: document.getElementById("window-counts"),
    windowTotal: document.getElementById("window-total"),
    windowCountSummary: document.getElementById("window-count-summary"),
    modelResults: document.getElementById("model-results"),
    fusionInformation: document.getElementById("fusion-information"),
  };
  const brainMotion = {
    button: document.getElementById("signal-motion-button"),
    videos: Array.from(document.querySelectorAll(".brain-reference-video")),
    paused: window.matchMedia("(prefers-reduced-motion: reduce)").matches,
  };

  function setBrainMotionPaused(paused) {
    brainMotion.paused = paused;
    brainMotion.button.setAttribute("aria-pressed", String(paused));
    brainMotion.button.textContent = paused ? "Play 3D animation" : "Pause 3D animation";
    syncBrainVideos();
  }

  function syncBrainVideos() {
    brainMotion.videos.forEach((video) => {
      const isHidden = Boolean(video.closest("[hidden]"));
      if (brainMotion.paused || isHidden) {
        video.pause();
        return;
      }
      const playback = video.play();
      if (playback && typeof playback.catch === "function") playback.catch(() => {});
    });
  }

  function showError(message) {
    elements.error.textContent = message;
    elements.error.hidden = false;
  }

  function clearError() {
    elements.error.textContent = "";
    elements.error.hidden = true;
  }

  function setFlowStep(step) {
    elements.flowSteps.forEach((item) => {
      const current = Number(item.dataset.flowStep);
      item.classList.toggle("active", current === step);
      item.classList.toggle("complete", current < step);
      if (current === step) item.setAttribute("aria-current", "step");
      else item.removeAttribute("aria-current");
    });
  }

  async function startLocalModules() {
    elements.startModulesButton.disabled = true;
    elements.moduleStartMessage.textContent = "Connecting to the local inference service…";
    const online = await refreshHealth();
    if (online) {
      elements.moduleStartPanel.hidden = true;
      elements.uploadPanel.hidden = false;
      elements.moduleStartMessage.textContent = "Backend online. Saved checkpoints will load when you run the first recording.";
      setFlowStep(2);
      syncBrainVideos();
    } else {
      elements.moduleStartMessage.textContent = "The local API is offline. Start it from the project Terminal, then try again.";
    }
    elements.startModulesButton.disabled = false;
  }

  function setLoading(isLoading) {
    const file = elements.fileInput.files && elements.fileInput.files[0];
    const isMatFile = Boolean(file && file.name.toLowerCase().endsWith(".mat"));
    elements.predictButton.disabled = isLoading || !isMatFile;
    elements.fileInput.disabled = isLoading;
    elements.results.setAttribute("aria-busy", String(isLoading));
    elements.predictButtonLabel.textContent = isLoading ? "Analyzing recording…" : "Run local analysis";
    if (elements.uploadStatus) {
      elements.uploadStatus.textContent = isLoading
        ? "Analyzing on the host laptop. The first analysis loads the saved models; keep this page open."
        : isMatFile
          ? "Ready to send this recording to the local inference service."
          : "Choose a .mat file to enable analysis.";
    }
  }

  function formatPercent(value) {
    const number = Number(value);
    return Number.isFinite(number) ? `${(number * 100).toFixed(1)}%` : "—";
  }

  function formatDecimal(value) {
    const number = Number(value);
    return Number.isFinite(number) ? number.toFixed(4) : "—";
  }

  function createElement(tag, className, text) {
    const element = document.createElement(tag);
    if (className) element.className = className;
    if (text !== undefined && text !== null) element.textContent = String(text);
    return element;
  }

  function validateResponse(payload) {
    if (!payload || payload.ok !== true || typeof payload.prediction !== "string") return false;
    if (!Number.isFinite(Number(payload.confidence))) return false;
    if (!Array.isArray(payload.class_order) || !payload.class_probabilities) return false;
    return CLASS_ORDER.every((name) => Number.isFinite(Number(payload.class_probabilities[name])));
  }

  function renderProbabilities(probabilities) {
    elements.probabilityList.replaceChildren();
    CLASS_ORDER.forEach((name) => {
      const value = Number(probabilities[name]);
      const row = createElement("div", "probability-row");
      const label = createElement("span", "probability-label", name);
      const track = createElement("div", "probability-track");
      track.setAttribute("role", "progressbar");
      track.setAttribute("aria-label", `${name} probability`);
      track.setAttribute("aria-valuemin", "0");
      track.setAttribute("aria-valuemax", "100");
      track.setAttribute("aria-valuenow", String(value * 100));
      const fill = createElement("span", "probability-fill");
      fill.style.width = `${Math.max(0, Math.min(100, value * 100))}%`;
      track.append(fill);
      const score = createElement("span", "probability-value", formatPercent(value));
      row.append(label, track, score);
      elements.probabilityList.append(row);
    });
  }

  function renderWindowCounts(counts) {
    elements.windowCounts.replaceChildren();
    const entries = [];
    if (counts && counts.raw !== undefined) entries.push(["Raw EEG windows", counts.raw]);
    if (counts && counts.stft !== undefined) entries.push(["STFT windows", counts.stft]);
    if (counts && counts.by_model && typeof counts.by_model === "object") {
      Object.entries(counts.by_model).forEach(([name, count]) => entries.push([name, count]));
    }
    if (entries.length === 0) {
      elements.windowCounts.append(createElement("p", "empty-data", "Window counts were not included in the response."));
    } else {
      entries.forEach(([label, value]) => {
        const card = createElement("div", "count-card");
        card.append(createElement("span", "count-label", label));
        card.append(createElement("span", "count-value", value));
        elements.windowCounts.append(card);
      });
    }
    const raw = Number(counts && counts.raw) || 0;
    const stft = Number(counts && counts.stft) || 0;
    const modelCounts = counts && counts.by_model && typeof counts.by_model === "object"
      ? Object.values(counts.by_model).map(Number).filter(Number.isFinite)
      : [];
    const total = Math.max(raw, stft, ...modelCounts, 0);
    elements.windowTotal.textContent = total ? String(total) : "—";
    elements.windowCountSummary.textContent = raw || stft
      ? `${raw} raw · ${stft} STFT windows`
      : total ? `${total} windows processed` : "Window counts unavailable";
    const timeline = document.querySelector(".timeline-track");
    if (timeline) timeline.setAttribute("aria-label", `${total || "Unknown number of"} analysis windows for this recording`);
  }

  function renderModelResults(models) {
    elements.modelResults.replaceChildren();
    const entries = models && typeof models === "object" ? Object.entries(models) : [];
    if (entries.length === 0) {
      elements.modelResults.append(createElement("p", "empty-data", "Per-model results were not included in the response."));
      return;
    }
    entries.forEach(([name, result]) => {
      const card = createElement("article", "model-card");
      card.append(createElement("span", "model-name", name));
      card.append(createElement("span", "model-prediction", result && result.prediction !== undefined ? result.prediction : "—"));
      const meta = createElement("div", "model-meta");
      if (result && result.confidence !== undefined) meta.append(createElement("span", "", `Confidence ${formatPercent(result.confidence)}`));
      if (result && result.window_count !== undefined) meta.append(createElement("span", "", `${result.window_count} windows`));
      card.append(meta);
      if (result && result.class_probabilities && typeof result.class_probabilities === "object") {
        const details = createElement("details", "model-details");
        details.append(createElement("summary", "", "Class probabilities"));
        const rows = createElement("div", "mini-probabilities");
        CLASS_ORDER.forEach((className) => {
          if (result.class_probabilities[className] === undefined) return;
          const row = createElement("div", "mini-probability");
          row.append(createElement("span", "", className));
          row.append(createElement("span", "", formatPercent(result.class_probabilities[className])));
          rows.append(row);
        });
        details.append(rows);
        card.append(details);
      }
      elements.modelResults.append(card);
    });
  }

  function renderFusionInformation(fusion) {
    elements.fusionInformation.replaceChildren();
    const labels = [
      ["weights", "Adaptive weights"],
      ["reliability_prior_weights", "Reliability prior weights"],
      ["certainty_gate_by_model", "Certainty gates"],
      ["normalized_entropy_by_model", "Normalized entropy"],
    ];
    let rendered = false;
    labels.forEach(([key, label]) => {
      const values = fusion && fusion[key];
      if (!values || typeof values !== "object") return;
      rendered = true;
      const card = createElement("section", "fusion-card");
      card.append(createElement("h3", "fusion-title", label));
      const rows = createElement("div", "fusion-rows");
      Object.entries(values).forEach(([name, value]) => {
        const row = createElement("div", "fusion-row");
        row.append(createElement("span", "", name));
        row.append(createElement("span", "", formatDecimal(value)));
        rows.append(row);
      });
      card.append(rows);
      elements.fusionInformation.append(card);
    });
    if (!rendered) {
      elements.fusionInformation.append(createElement("p", "empty-data", "Adaptive-fusion details were not included in the response."));
    }
  }

  function renderPrediction(payload, file) {
    if (!validateResponse(payload)) {
      throw new Error("The backend returned an incomplete prediction response.");
    }
    const modelCount = payload.models && typeof payload.models === "object" ? Object.keys(payload.models).length : 0;
    elements.resultFile.textContent = file.name;
    elements.resultRecordMeta.textContent = modelCount
      ? `Analyzed by ${modelCount} local models · ${payload.confidence_type || "confidence score"}`
      : "Prediction returned by the local inference service.";
    elements.resultEmotion.textContent = payload.prediction;
    elements.resultConfidence.textContent = formatPercent(payload.confidence);
    elements.confidenceDescription.textContent = payload.confidence_description || "Frozen adaptive confidence score.";
    renderProbabilities(payload.class_probabilities);
    renderWindowCounts(payload.window_counts);
    renderModelResults(payload.models);
    renderFusionInformation(payload.adaptive_fusion);
    elements.uploadPanel.hidden = true;
    elements.results.hidden = false;
    setFlowStep(3);
    syncBrainVideos();
  }

  async function refreshHealth() {
    try {
      const response = await fetch(`${API_BASE_URL}/health`, { method: "GET" });
      const payload = await response.json();
      const online = response.ok && payload.ok === true && payload.status === "ok";
      elements.healthText.textContent = online ? "Backend online" : "Backend unavailable";
      elements.healthIndicator.className = `local-dot ${online ? "online" : "offline"}`;
      elements.moduleApiStatus.textContent = online ? "API reachable · models load on analysis" : "Backend offline";
      elements.moduleApiStatus.className = `module-api-status ${online ? "online" : "offline"}`;
      return online;
    } catch (_error) {
      elements.healthText.textContent = "Backend unavailable";
      elements.healthIndicator.className = "local-dot offline";
      elements.moduleApiStatus.textContent = "Backend offline";
      elements.moduleApiStatus.className = "module-api-status offline";
      return false;
    }
  }

  function validateFile(file) {
    if (!file) {
      showError("Select a .mat EEG recording before starting a prediction.");
      return false;
    }
    if (!file.name.toLowerCase().endsWith(".mat")) {
      showError("The selected file is not a .mat recording. Choose a MATLAB EEG file.");
      return false;
    }
    return true;
  }

  async function predict() {
    clearError();
    const file = elements.fileInput.files && elements.fileInput.files[0];
    if (!validateFile(file)) return;

    const formData = new FormData();
    formData.append("file", file, file.name);
    elements.results.hidden = true;
    setLoading(true);
    const controller = new AbortController();
    const timeout = window.setTimeout(() => controller.abort(), 5 * 60 * 1000);
    try {
      const response = await fetch(`${API_BASE_URL}/predict`, {
        method: "POST",
        body: formData,
        signal: controller.signal,
      });
      let payload;
      try {
        payload = await response.json();
      } catch (_error) {
        throw new Error("The backend returned a response that could not be read.");
      }
      if (!response.ok || payload.ok !== true) {
        throw new Error(payload && payload.error && payload.error.message
          ? payload.error.message
          : "The recording could not be analyzed. Check the EEG file and try again.");
      }
      renderPrediction(payload, file);
      await refreshHealth();
    } catch (error) {
      if (error && error.name === "AbortError") {
        showError("The analysis took too long. Check that the backend is running, then try again.");
      } else if (error instanceof TypeError) {
        showError("The backend is unavailable. Start the backend service and try again.");
      } else {
        showError(error && error.message ? error.message : "The prediction could not be completed.");
      }
    } finally {
      window.clearTimeout(timeout);
      setLoading(false);
    }
  }

  function reset() {
    elements.fileInput.value = "";
    elements.fileName.textContent = "No file selected";
    elements.results.hidden = true;
    elements.results.setAttribute("aria-busy", "false");
    elements.uploadPanel.hidden = false;
    elements.moduleStartPanel.hidden = true;
    setFlowStep(2);
    clearError();
    setLoading(false);
    syncBrainVideos();
  }

  elements.fileInput.addEventListener("change", () => {
    clearError();
    const file = elements.fileInput.files && elements.fileInput.files[0];
    elements.fileName.textContent = file ? file.name : "No file selected";
    elements.predictButton.disabled = !file || !file.name.toLowerCase().endsWith(".mat");
    if (elements.uploadStatus) {
      elements.uploadStatus.textContent = file && file.name.toLowerCase().endsWith(".mat")
        ? "Ready to send this recording to the local inference service."
        : "Choose a .mat file to enable analysis.";
    }
    if (file && !file.name.toLowerCase().endsWith(".mat")) {
      showError("The selected file is not a .mat recording. Choose a MATLAB EEG file.");
    }
  });
  elements.predictButton.addEventListener("click", predict);
  elements.resetButton.addEventListener("click", reset);
  elements.resultsResetButton.addEventListener("click", reset);
  elements.startModulesButton.addEventListener("click", startLocalModules);
  if (brainMotion.paused) setBrainMotionPaused(true);
  brainMotion.button.addEventListener("click", () => {
    setBrainMotionPaused(brainMotion.button.getAttribute("aria-pressed") !== "true");
  });
  elements.uploadZone.addEventListener("dragover", (event) => {
    event.preventDefault();
    elements.uploadZone.classList.add("drag-active");
  });
  elements.uploadZone.addEventListener("dragleave", () => elements.uploadZone.classList.remove("drag-active"));
  elements.uploadZone.addEventListener("drop", (event) => {
    event.preventDefault();
    elements.uploadZone.classList.remove("drag-active");
    if (!event.dataTransfer || !event.dataTransfer.files || !event.dataTransfer.files.length) return;
    elements.fileInput.files = event.dataTransfer.files;
    elements.fileInput.dispatchEvent(new Event("change", { bubbles: true }));
  });

  refreshHealth();
  window.setInterval(refreshHealth, 10000);
  syncBrainVideos();
  setFlowStep(1);
})();
