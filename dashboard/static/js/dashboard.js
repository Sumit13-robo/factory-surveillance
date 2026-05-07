/**
 * Factory AI Dashboard — Real-time data fetching and UI updates.
 *
 * Polls /api/stats and /api/incidents endpoints every 2 seconds
 * and updates the dashboard UI with live GPU stats and incidents.
 */

// === Config ===
const POLL_INTERVAL_MS = 2000;

// === State ===
let totalAlerts = 0;
let lastIncidentCount = 0;

// === DOM Elements ===
const elements = {
    fpsValue: document.getElementById("fps-value"),
    vramValue: document.getElementById("vram-value"),
    gpuUtilValue: document.getElementById("gpu-util-value"),
    alertCount: document.getElementById("alert-count"),
    detectionCount: document.getElementById("detection-count"),
    trackingCount: document.getElementById("tracking-count"),
    gpuName: document.getElementById("gpu-name"),
    vramBar: document.getElementById("vram-bar"),
    vramDetail: document.getElementById("vram-detail"),
    gpuBar: document.getElementById("gpu-bar"),
    gpuDetail: document.getElementById("gpu-detail"),
    gpuTemp: document.getElementById("gpu-temp"),
    gpuPower: document.getElementById("gpu-power"),
    inferenceLatency: document.getElementById("inference-latency"),
    ppeCount: document.getElementById("ppe-count"),
    hazardCount: document.getElementById("hazard-count"),
    unsafeCount: document.getElementById("unsafe-count"),
    machineryCount: document.getElementById("machinery-count"),
    incidentTbody: document.getElementById("incident-tbody"),
};

// === Fetch Stats ===
async function fetchStats() {
    try {
        const resp = await fetch("/api/stats");
        if (!resp.ok) return;
        const data = await resp.json();
        updateStatsUI(data);
    } catch (err) {
        console.debug("Stats fetch error:", err);
    }
}

function updateStatsUI(data) {
    // Header stats
    if (data.fps !== undefined) {
        elements.fpsValue.textContent = data.fps.toFixed(1);
    }

    // GPU stats
    const gpu = data.gpu || {};
    if (gpu.device_name) elements.gpuName.textContent = gpu.device_name;

    if (gpu.vram_used_mb !== undefined) {
        const vramPct = gpu.vram_percent || 0;
        elements.vramValue.textContent = `${vramPct.toFixed(0)}%`;
        elements.vramBar.style.width = `${vramPct}%`;
        elements.vramDetail.textContent = `${gpu.vram_used_mb.toFixed(0)}/${gpu.vram_total_mb.toFixed(0)} MB`;

        // Color coding
        if (vramPct > 90) {
            elements.vramBar.style.background = "linear-gradient(90deg, #ef4444, #dc2626)";
        } else if (vramPct > 70) {
            elements.vramBar.style.background = "linear-gradient(90deg, #f59e0b, #d97706)";
        }
    }

    if (gpu.gpu_utilization !== undefined) {
        elements.gpuUtilValue.textContent = `${gpu.gpu_utilization}%`;
        elements.gpuBar.style.width = `${gpu.gpu_utilization}%`;
        elements.gpuDetail.textContent = `${gpu.gpu_utilization}%`;
    }

    if (gpu.temperature !== undefined) {
        elements.gpuTemp.textContent = `${gpu.temperature}°C`;
        elements.gpuTemp.style.color = gpu.temperature > 80 ? "#ef4444" : gpu.temperature > 65 ? "#f59e0b" : "#10b981";
    }

    if (gpu.power_draw_w !== undefined) {
        elements.gpuPower.textContent = `${gpu.power_draw_w.toFixed(1)}W`;
    }

    // Inference latency
    if (data.inference_ms !== undefined) {
        elements.inferenceLatency.textContent = `${data.inference_ms.toFixed(1)}ms`;
    }

    // Detection counts
    if (data.detection_count !== undefined) {
        elements.detectionCount.textContent = `${data.detection_count} detections`;
    }
    if (data.tracking_count !== undefined) {
        elements.trackingCount.textContent = `${data.tracking_count} tracked`;
    }

    // Category counts
    const cats = data.categories || {};
    elements.ppeCount.textContent = cats.ppe_violation || 0;
    elements.hazardCount.textContent = cats.hazard || 0;
    elements.unsafeCount.textContent = cats.unsafe_behavior || 0;
    elements.machineryCount.textContent = cats.machinery_risk || 0;

    // Alert count
    if (data.alert_count !== undefined) {
        totalAlerts = data.alert_count;
        elements.alertCount.textContent = totalAlerts;
    }
}

// === Fetch Incidents ===
async function fetchIncidents() {
    try {
        const resp = await fetch("/api/incidents?limit=50");
        if (!resp.ok) return;
        const incidents = await resp.json();
        updateIncidentsUI(incidents);
    } catch (err) {
        console.debug("Incidents fetch error:", err);
    }
}

function updateIncidentsUI(incidents) {
    if (incidents.length === 0) return;
    if (incidents.length === lastIncidentCount) return;
    lastIncidentCount = incidents.length;

    const rows = incidents
        .reverse()
        .map(inc => {
            const time = inc.timestamp
                ? new Date(inc.timestamp).toLocaleTimeString()
                : "--";
            const typeClass = (inc.type || "").replace(/\s/g, "_");
            return `
                <tr>
                    <td>${time}</td>
                    <td><span class="type-badge ${typeClass}">${inc.type || "--"}</span></td>
                    <td>${inc.class_name || "--"}</td>
                    <td>${inc.confidence ? (inc.confidence * 100).toFixed(0) + "%" : "--"}</td>
                    <td>${inc.camera || "--"}</td>
                    <td>${inc.message || "--"}</td>
                </tr>
            `;
        })
        .join("");

    elements.incidentTbody.innerHTML = rows;
}

function clearIncidents() {
    elements.incidentTbody.innerHTML = `
        <tr class="empty-row">
            <td colspan="6">No incidents recorded yet</td>
        </tr>
    `;
    lastIncidentCount = 0;
}

// === Polling Loop ===
function startPolling() {
    setInterval(() => {
        fetchStats();
        fetchIncidents();
    }, POLL_INTERVAL_MS);

    // Initial fetch
    fetchStats();
    fetchIncidents();
}

// === Start ===
document.addEventListener("DOMContentLoaded", startPolling);
