// ─── Configuration ───
const API_URL = "https://skin-cancer-backend-izl4.onrender.com";
// ─── DOM Elements ───
const dropZone = document.getElementById("dropZone");
const fileInput = document.getElementById("fileInput");
const uploadContent = document.getElementById("uploadContent");
const previewContainer = document.getElementById("previewContainer");
const previewImage = document.getElementById("previewImage");
const removeBtn = document.getElementById("removeBtn");
const predictBtn = document.getElementById("predictBtn");
const resultsSection = document.getElementById("resultsSection");
const mainResult = document.getElementById("mainResult");
const resultIcon = document.getElementById("resultIcon");
const resultLabel = document.getElementById("resultLabel");
const resultConfidence = document.getElementById("resultConfidence");
const riskBadge = document.getElementById("riskBadge");
const recommendationBox = document.getElementById("recommendationBox");
const recommendationText = document.getElementById("recommendationText");
const probabilityBars = document.getElementById("probabilityBars");
const loadingOverlay = document.getElementById("loadingOverlay");
const errorMessage = document.getElementById("errorMessage");
const errorText = document.getElementById("errorText");
const modelDetails = document.getElementById("modelDetails");
const newAnalysisBtn = document.getElementById("newAnalysisBtn");
const saveResultBtn = document.getElementById("saveResultBtn");

let selectedFile = null;
let currentResults = null;

// ─── Initialize ───
document.addEventListener("DOMContentLoaded", () => {
    fetchModelInfo();
    setupEventListeners();
});

// ─── Event Listeners ───
function setupEventListeners() {
    // File input
    fileInput.addEventListener("change", handleFileSelect);

    // Drag & drop
    dropZone.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropZone.classList.add("drag-over");
    });

    dropZone.addEventListener("dragleave", () => {
        dropZone.classList.remove("drag-over");
    });

    dropZone.addEventListener("drop", (e) => {
        e.preventDefault();
        dropZone.classList.remove("drag-over");
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFile(files[0]);
        }
    });

    dropZone.addEventListener("click", (e) => {
        if (e.target === dropZone || uploadContent.contains(e.target)) {
            fileInput.click();
        }
    });

    // Buttons
    removeBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        resetUpload();
    });

    predictBtn.addEventListener("click", handlePredict);
    newAnalysisBtn.addEventListener("click", resetUpload);
    saveResultBtn.addEventListener("click", saveResults);
}

// ─── File Handling ───
function handleFileSelect(e) {
    const file = e.target.files[0];
    if (file) handleFile(file);
}

function handleFile(file) {
    // Validate file type
    const validTypes = ["image/jpeg", "image/jpg", "image/png", "image/webp"];
    if (!validTypes.includes(file.type.toLowerCase())) {
        showError("Invalid file type. Please upload JPG, PNG, or WEBP images only.");
        return;
    }

    // Validate file size (10MB max)
    const maxSize = 10 * 1024 * 1024;
    if (file.size > maxSize) {
        showError(`File too large (${(file.size / 1024 / 1024).toFixed(2)}MB). Maximum size is 10MB.`);
        return;
    }

    selectedFile = file;
    hideError();

    // Show preview
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImage.src = e.target.result;
        uploadContent.style.display = "none";
        previewContainer.style.display = "flex";
        predictBtn.disabled = false;
    };
    reader.onerror = () => {
        showError("Failed to read the image file.");
    };
    reader.readAsDataURL(file);
}

function resetUpload() {
    selectedFile = null;
    currentResults = null;
    fileInput.value = "";
    previewImage.src = "";
    uploadContent.style.display = "block";
    previewContainer.style.display = "none";
    predictBtn.disabled = true;
    resultsSection.style.display = "none";
    hideError();
    
    // Scroll to top
    window.scrollTo({ top: 0, behavior: "smooth" });
}

// ─── Prediction ───
async function handlePredict() {
    if (!selectedFile) return;

    // Show loading state
    loadingOverlay.style.display = "block";
    resultsSection.style.display = "none";
    hideError();
    predictBtn.disabled = true;

    try {
        // Create form data
        const formData = new FormData();
        formData.append("file", selectedFile);

        // Make API request
        const response = await fetch(`${API_URL}/predict`, {
            method: "POST",
            body: formData,
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || `Server error: ${response.status}`);
        }

        currentResults = data;
        displayResults(data);

    } catch (error) {
        console.error("Prediction error:", error);
        showError(
            error.message || 
            "Unable to analyze the image. Please check your connection and try again."
        );
    } finally {
        loadingOverlay.style.display = "none";
        predictBtn.disabled = false;
    }
}

// ─── Display Results ───
function displayResults(data) {
    resultsSection.style.display = "block";

    const isBenign = data.prediction === "Benign";
    const predictionClass = isBenign ? "benign" : "malignant";

    // Update main result card
    mainResult.className = `main-result ${predictionClass}`;
    
    // Update icon
    resultIcon.className = `result-icon ${predictionClass}`;
    resultIcon.innerHTML = isBenign 
        ? '<i class="fas fa-check-circle"></i>' 
        : '<i class="fas fa-exclamation-triangle"></i>';

    // Update label
    resultLabel.textContent = data.prediction;
    resultLabel.className = `result-label ${predictionClass}`;

    // Update confidence
    resultConfidence.textContent = `Confidence: ${data.confidence}%`;

    // Update risk badge
    const riskLevel = data.risk_level.toLowerCase().replace(/\s+/g, "-");
    let badgeClass = "low";
    if (riskLevel.includes("high")) badgeClass = "high";
    else if (riskLevel.includes("moderate")) badgeClass = "moderate";
    
    riskBadge.textContent = data.risk_level;
    riskBadge.className = `risk-badge ${badgeClass}`;

    // Update recommendation
    recommendationText.textContent = data.recommendation;

    // Update probability bars
    displayProbabilityBars(data.probabilities);

    // Scroll to results
    setTimeout(() => {
        resultsSection.scrollIntoView({ behavior: "smooth", block: "start" });
    }, 100);
}

function displayProbabilityBars(probabilities) {
    probabilityBars.innerHTML = "";

    // Sort to show higher probability first
    const sorted = Object.entries(probabilities).sort((a, b) => b[1] - a[1]);

    sorted.forEach(([label, probability]) => {
        const labelClass = label.toLowerCase();
        
        const item = document.createElement("div");
        item.className = "prob-item";
        item.innerHTML = `
            <span class="prob-label ${labelClass}">${label}</span>
            <div class="prob-bar-container">
                <div class="prob-bar ${labelClass}" style="width: 0%">
                    ${probability >= 10 ? probability.toFixed(1) + '%' : ''}
                </div>
            </div>
            <span class="prob-value">${probability.toFixed(1)}%</span>
        `;
        probabilityBars.appendChild(item);

        // Animate bar
        setTimeout(() => {
            item.querySelector(".prob-bar").style.width = `${probability}%`;
        }, 100);
    });
}

// ─── Save Results ───
function saveResults() {
    if (!currentResults) return;

    const timestamp = new Date().toISOString().replace(/[:.]/g, "-");
    const filename = `skin-cancer-analysis-${timestamp}.txt`;

    const content = `
═══════════════════════════════════════════════════════
SKIN CANCER DETECTION - ANALYSIS RESULTS
═══════════════════════════════════════════════════════

Date & Time: ${new Date().toLocaleString()}

PREDICTION
────────────────────────────────────────────────────────
Classification: ${currentResults.prediction}
Confidence: ${currentResults.confidence}%
Risk Level: ${currentResults.risk_level}

PROBABILITY BREAKDOWN
────────────────────────────────────────────────────────
Benign: ${currentResults.probabilities.Benign}%
Malignant: ${currentResults.probabilities.Malignant}%

MEDICAL RECOMMENDATION
────────────────────────────────────────────────────────
${currentResults.recommendation}

IMPORTANT DISCLAIMER
────────────────────────────────────────────────────────
${currentResults.disclaimer}

This AI analysis should be used as a preliminary screening
tool only. Always consult a qualified dermatologist for
professional medical diagnosis and treatment.

═══════════════════════════════════════════════════════
Generated by Skin Cancer Detector AI
═══════════════════════════════════════════════════════
    `.trim();

    // Create download
    const blob = new Blob([content], { type: "text/plain" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
}

// ─── Model Info ───
async function fetchModelInfo() {
    try {
        const response = await fetch(`${API_URL}/model-info`);
        const data = await response.json();

        modelDetails.innerHTML = `
            <div class="model-detail">
                <span>Model Status</span>
                <span style="color: var(--success)">● Active</span>
            </div>
            <div class="model-detail">
                <span>Input Size</span>
                <span>${data.input_size}</span>
            </div>
            <div class="model-detail">
                <span>Parameters</span>
                <span>${(data.total_parameters / 1e6).toFixed(2)}M</span>
            </div>
            <div class="model-detail">
                <span>Classes</span>
                <span>${data.classes.join(", ")}</span>
            </div>
            <div class="model-detail">
                <span>Model Type</span>
                <span>${data.model_type}</span>
            </div>
        `;
    } catch (error) {
        modelDetails.innerHTML = `
            <div class="model-detail">
                <span>Model Status</span>
                <span style="color: var(--danger)">● Offline</span>
            </div>
            <div class="model-detail">
                <span colspan="2">Could not connect to backend server</span>
            </div>
        `;
    }
}

// ─── Error Handling ───
function showError(message) {
    errorText.textContent = message;
    errorMessage.style.display = "flex";
    errorMessage.scrollIntoView({ behavior: "smooth", block: "center" });
}

function hideError() {
    errorMessage.style.display = "none";
}

// ─── Utility: Handle offline/online ───
window.addEventListener("online", () => {
    hideError();
    fetchModelInfo();
});

window.addEventListener("offline", () => {
    showError("No internet connection. Please check your network.");
});