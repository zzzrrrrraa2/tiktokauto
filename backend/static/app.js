let currentVideoId = null;
let currentRoi = null;
let pollingInterval = null;

const uploadArea = document.getElementById("upload-area");
const uploadPlaceholder = document.getElementById("upload-placeholder");
const uploadProgress = document.getElementById("upload-progress");
const videoInput = document.getElementById("video-input");
const roiSection = document.getElementById("roi-section");
const roiContainer = document.getElementById("roi-container");
const roiBox = document.getElementById("roi-box");
const roiCoords = document.getElementById("roi-coords");
const roiReset = document.getElementById("roi-reset");
const btnStart = document.getElementById("btn-start");
const progressSection = document.getElementById("progress-section");
const progressBar = document.getElementById("progress-bar");
const progressText = document.getElementById("progress-text");
const progressStatus = document.getElementById("progress-status");
const clipList = document.getElementById("clip-list");
const btnCancel = document.getElementById("btn-cancel");
const resultSection = document.getElementById("result-section");
const resultVideo = document.getElementById("result-video");
const btnDownload = document.getElementById("btn-download");
const btnNewVideo = document.getElementById("btn-new-video");
const historyList = document.getElementById("history-list");
const previewVideo = document.getElementById("preview-video");
const videoTimeline = document.getElementById("video-timeline");
const videoTime = document.getElementById("video-time");
const btnPlayPause = document.getElementById("btn-play-pause");

uploadArea.addEventListener("click", () => videoInput.click());
videoInput.addEventListener("change", handleUpload);

uploadArea.addEventListener("dragover", (e) => {
    e.preventDefault();
    uploadArea.style.borderColor = "#5a5aff";
});
uploadArea.addEventListener("dragleave", () => {
    uploadArea.style.borderColor = "#3a3a3a";
});
uploadArea.addEventListener("drop", (e) => {
    e.preventDefault();
    uploadArea.style.borderColor = "#3a3a3a";
    const file = e.dataTransfer.files[0];
    if (file) uploadFile(file);
});

roiReset.addEventListener("click", resetRoi);
btnStart.addEventListener("click", startProcessing);
btnCancel.addEventListener("click", cancelProcessing);
btnNewVideo.addEventListener("click", resetAll);

let roiDrawing = false;
let roiStart = { x: 0, y: 0 };

function getVideoScales() {
    const videoRect = previewVideo.getBoundingClientRect();
    return {
        scaleX: previewVideo.videoWidth / videoRect.width,
        scaleY: previewVideo.videoHeight / videoRect.height,
        displayScaleX: videoRect.width / previewVideo.videoWidth,
        displayScaleY: videoRect.height / previewVideo.videoHeight,
        left: videoRect.left,
        top: videoRect.top
    };
}

previewVideo.addEventListener("mousedown", (e) => {
    const s = getVideoScales();
    roiDrawing = true;
    roiStart = {
        x: (e.clientX - s.left) * s.scaleX,
        y: (e.clientY - s.top) * s.scaleY
    };
    roiBox.style.display = "block";
    roiBox.style.left = "0px";
    roiBox.style.top = "0px";
    roiBox.style.width = "0px";
    roiBox.style.height = "0px";
});

previewVideo.addEventListener("mousemove", (e) => {
    if (!roiDrawing) return;
    const s = getVideoScales();

    const currentX = (e.clientX - s.left) * s.scaleX;
    const currentY = (e.clientY - s.top) * s.scaleY;

    const x = Math.min(roiStart.x, currentX);
    const y = Math.min(roiStart.y, currentY);
    const w = Math.abs(currentX - roiStart.x);
    const h = Math.abs(currentY - roiStart.y);

    roiBox.style.left = (x * s.displayScaleX) + "px";
    roiBox.style.top = (y * s.displayScaleY) + "px";
    roiBox.style.width = (w * s.displayScaleX) + "px";
    roiBox.style.height = (h * s.displayScaleY) + "px";
});

previewVideo.addEventListener("mouseup", (e) => {
    if (!roiDrawing) return;
    roiDrawing = false;

    const s = getVideoScales();

    const currentX = (e.clientX - s.left) * s.scaleX;
    const currentY = (e.clientY - s.top) * s.scaleY;

    const x = Math.round(Math.min(roiStart.x, currentX));
    const y = Math.round(Math.min(roiStart.y, currentY));
    const w = Math.round(Math.abs(currentX - roiStart.x));
    const h = Math.round(Math.abs(currentY - roiStart.y));

    if (w < 10 || h < 10) {
        roiBox.style.display = "none";
        return;
    }

    currentRoi = { x, y, w, h };
    roiCoords.innerHTML = `x=${x} y=${y} w=${w} h=${h}`;
    btnStart.disabled = false;

    fetch(`/api/video/${currentVideoId}/roi`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(currentRoi)
    });
});

videoTimeline.addEventListener("input", () => {
    const t = parseFloat(videoTimeline.value);
    previewVideo.currentTime = t;
});

previewVideo.addEventListener("timeupdate", () => {
    if (previewVideo.duration) {
        videoTimeline.value = previewVideo.currentTime;
        videoTime.textContent = formatTime(previewVideo.currentTime) + " / " + formatTime(previewVideo.duration);
    }
});

previewVideo.addEventListener("loadedmetadata", () => {
    videoTimeline.max = previewVideo.duration;
    videoTime.textContent = "0:00 / " + formatTime(previewVideo.duration);
});

btnPlayPause.addEventListener("click", () => {
    if (previewVideo.paused) {
        previewVideo.play();
        btnPlayPause.textContent = "⏸";
    } else {
        previewVideo.pause();
        btnPlayPause.textContent = "▶";
    }
});

previewVideo.addEventListener("play", () => { btnPlayPause.textContent = "⏸"; });
previewVideo.addEventListener("pause", () => { btnPlayPause.textContent = "▶"; });

function formatTime(seconds) {
    const m = Math.floor(seconds / 60);
    const s = Math.floor(seconds % 60);
    return m + ":" + s.toString().padStart(2, "0");
}

async function handleUpload(e) {
    const file = e.target.files[0];
    if (file) await uploadFile(file);
}

async function uploadFile(file) {
    uploadPlaceholder.classList.add("hidden");
    uploadProgress.classList.remove("hidden");

    const formData = new FormData();
    formData.append("video", file);

    try {
        const resp = await fetch("/api/upload", { method: "POST", body: formData });
        const data = await resp.json();

        if (data.error) {
            alert(data.error);
            uploadProgress.classList.add("hidden");
            uploadPlaceholder.classList.remove("hidden");
            return;
        }

        currentVideoId = data.video_id;
        previewVideo.src = data.video_url;
        roiSection.classList.remove("hidden");
        resetRoi();

        document.getElementById("upload-section").querySelector("h2").textContent =
            `1. Uploaded: ${data.filename} (${data.width}x${data.height}, ${data.duration.toFixed(1)}s)`;
    } catch (err) {
        alert("Upload failed: " + err.message);
        uploadProgress.classList.add("hidden");
        uploadPlaceholder.classList.remove("hidden");
    }
}

function resetRoi() {
    currentRoi = null;
    roiBox.style.display = "none";
    roiCoords.innerHTML = "Not selected";
    btnStart.disabled = true;
}

async function startProcessing() {
    if (!currentVideoId || !currentRoi) return;

    roiSection.classList.add("hidden");
    progressSection.classList.remove("hidden");
    resultSection.classList.add("hidden");
    btnCancel.classList.remove("hidden");
    clipList.innerHTML = "";
    progressBar.style.width = "0%";
    progressText.textContent = "Starting...";
    progressStatus.textContent = "Sending to GPU...";

    try {
        const resp = await fetch(`/api/video/${currentVideoId}/start`, { method: "POST" });
        const data = await resp.json();
        if (data.error) {
            alert(data.error);
            return;
        }
        startPolling();
    } catch (err) {
        alert("Failed to start: " + err.message);
    }
}

function startPolling() {
    if (pollingInterval) clearInterval(pollingInterval);
    pollingInterval = setInterval(pollStatus, 1000);
}

async function pollStatus() {
    if (!currentVideoId) return;

    try {
        const resp = await fetch(`/api/video/${currentVideoId}/status`);
        const data = await resp.json();

        const total = data.total || 0;
        const completed = data.completed || 0;

        const pct = total > 0 ? Math.round((completed / total) * 100) : 0;
        progressBar.style.width = pct + "%";
        progressText.textContent = `${completed} / ${total} clips`;
        progressStatus.textContent = data.status;

        if (data.status === "completed") {
            clearInterval(pollingInterval);
            pollingInterval = null;
            btnCancel.classList.add("hidden");
            showResult();
        } else if (data.status === "failed") {
            clearInterval(pollingInterval);
            pollingInterval = null;
            btnCancel.classList.add("hidden");
            progressStatus.textContent = "Failed - processing did not complete";
        }
    } catch (err) {
        console.error("Poll error:", err);
    }
}

async function showResult() {
    progressSection.classList.add("hidden");
    resultSection.classList.remove("hidden");

    const finalPath = `/data/final/${currentVideoId}_final.mp4`;
    resultVideo.src = finalPath;
    btnDownload.href = finalPath;
    btnDownload.download = `${currentVideoId}_final.mp4`;

    loadHistory();
}

async function cancelProcessing() {
    if (!currentVideoId) return;
    await fetch(`/api/video/${currentVideoId}/cancel`, { method: "POST" });
    if (pollingInterval) {
        clearInterval(pollingInterval);
        pollingInterval = null;
    }
    btnCancel.classList.add("hidden");
    progressStatus.textContent = "Cancelled";
}

function resetAll() {
    currentVideoId = null;
    currentRoi = null;
    if (pollingInterval) clearInterval(pollingInterval);
    pollingInterval = null;

    roiSection.classList.add("hidden");
    progressSection.classList.add("hidden");
    resultSection.classList.add("hidden");
    uploadProgress.classList.add("hidden");
    uploadPlaceholder.classList.remove("hidden");
    document.getElementById("upload-section").querySelector("h2").textContent = "1. Upload Video";
    videoInput.value = "";
    previewVideo.src = "";
    loadHistory();
}

async function loadHistory() {
    try {
        const resp = await fetch("/api/videos");
        const videos = await resp.json();

        if (videos.length === 0) {
            historyList.innerHTML = '<p class="empty">No videos processed yet</p>';
            return;
        }

        historyList.innerHTML = videos.map(v => `
            <div class="history-item">
                <div>
                    <div class="name">${v.original_filename}</div>
                    <div class="meta">${v.width}x${v.height} · ${v.duration.toFixed(1)}s</div>
                </div>
                <span class="status-badge ${v.status}">${v.status}</span>
            </div>
        `).join("");
    } catch (err) {
        console.error("History error:", err);
    }
}

loadHistory();
