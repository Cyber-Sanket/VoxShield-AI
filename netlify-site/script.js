/**
 * VoxShield AI - Cyber Security Landing Page Script
 * Team: Ghost Protocol | Domain: AI & Cyber Security
 */

// ==========================================================================
// 1. CONFIGURATION
// Configure your Streamlit application URL here.
// For local demo: "http://localhost:8501"
// For cloud deployment: change to your deployed domain, e.g. "https://voxshield-ai.streamlit.app"
// ==========================================================================
const CONFIG = {
    DEFAULT_STREAMLIT_URL: "http://localhost:8501",
    GITHUB_REPO_URL: "https://github.com/Cyber-Sanket/VoxShield-AI"
};

// ==========================================================================
// 2. ENDPOINT MANAGEMENT & STORAGE
// ==========================================================================
function getActiveEndpoint() {
    const saved = localStorage.getItem("voxshield_app_url");
    return saved && saved.trim() !== "" ? saved.trim() : CONFIG.DEFAULT_STREAMLIT_URL;
}

function setActiveEndpoint(url) {
    if (url && url.trim() !== "") {
        localStorage.setItem("voxshield_app_url", url.trim());
    } else {
        localStorage.removeItem("voxshield_app_url");
    }
    updateAllLaunchButtons();
}

function updateAllLaunchButtons() {
    const endpoint = getActiveEndpoint();
    const buttons = document.querySelectorAll("[data-launch-btn]");
    buttons.forEach(btn => {
        btn.setAttribute("href", endpoint);
        btn.setAttribute("target", "_blank");
        btn.setAttribute("rel", "noopener noreferrer");
    });

    // Update displays
    const displayEndpoint = document.getElementById("displayEndpoint");
    if (displayEndpoint) displayEndpoint.textContent = endpoint;

    const activeEndpointText = document.querySelector(".active-endpoint-text");
    if (activeEndpointText) activeEndpointText.textContent = endpoint;

    const appUrlInput = document.getElementById("appUrlInput");
    if (appUrlInput) appUrlInput.value = endpoint;
}

// ==========================================================================
// 3. MODAL LOGIC FOR CHANGING ENDPOINT
// ==========================================================================
function setupModal() {
    const modal = document.getElementById("configModal");
    const editBtn = document.getElementById("editEndpointBtn");
    const closeBtn = document.getElementById("closeModalBtn");
    const saveBtn = document.getElementById("saveEndpointBtn");
    const resetBtn = document.getElementById("resetEndpointBtn");
    const input = document.getElementById("appUrlInput");

    if (!modal) return;

    editBtn?.addEventListener("click", () => {
        input.value = getActiveEndpoint();
        modal.classList.add("active");
    });

    closeBtn?.addEventListener("click", () => {
        modal.classList.remove("active");
    });

    modal.addEventListener("click", (e) => {
        if (e.target === modal) modal.classList.remove("active");
    });

    saveBtn?.addEventListener("click", () => {
        const newUrl = input.value.trim();
        if (newUrl) {
            setActiveEndpoint(newUrl);
            modal.classList.remove("active");
        }
    });

    resetBtn?.addEventListener("click", () => {
        setActiveEndpoint(CONFIG.DEFAULT_STREAMLIT_URL);
        modal.classList.remove("active");
    });
}

// ==========================================================================
// 4. INTERACTIVE CANVAS AUDIO SPECTRUM & WAVEFORM SIMULATOR
// ==========================================================================
class AudioVisualizerSimulator {
    constructor(canvasId) {
        this.canvas = document.getElementById(canvasId);
        if (!this.canvas) return;
        this.ctx = this.canvas.getContext("2d");
        this.currentMode = "human"; // "human" or "synth"
        this.animationId = null;
        this.phase = 0;
        this.numBars = 64;

        this.initResize();
        this.bindEvents();
        this.start();
    }

    initResize() {
        const resize = () => {
            const rect = this.canvas.parentElement.getBoundingClientRect();
            this.canvas.width = rect.width;
            this.canvas.height = 240;
        };
        resize();
        window.addEventListener("resize", resize);
    }

    bindEvents() {
        const btnHuman = document.getElementById("btnPresetHuman");
        const btnSynth = document.getElementById("btnPresetSynth");
        const tag = document.getElementById("simModeTag");
        const centroidEl = document.getElementById("metricCentroid");
        const rolloffEl = document.getElementById("metricRolloff");
        const flatnessEl = document.getElementById("metricFlatness");
        const insightBox = document.getElementById("insightBox");

        btnHuman?.addEventListener("click", () => {
            this.currentMode = "human";
            btnHuman.classList.add("active");
            btnSynth.classList.remove("active");
            if (tag) tag.textContent = "MODE: GENUINE VOICE";
            if (centroidEl) centroidEl.textContent = "2,140 Hz";
            if (rolloffEl) rolloffEl.textContent = "5,820 Hz";
            if (flatnessEl) flatnessEl.textContent = "0.0124";
            if (insightBox) insightBox.innerHTML = `<strong>Acoustic Insight:</strong> Natural harmonic vibrato, fluid pitch drift, and continuous energy in high frequencies (>4 kHz).`;
        });

        btnSynth?.addEventListener("click", () => {
            this.currentMode = "synth";
            btnSynth.classList.add("active");
            btnHuman.classList.remove("active");
            if (tag) tag.textContent = "MODE: SYNTHETIC VOCODER";
            if (centroidEl) centroidEl.textContent = "1,410 Hz (Low)";
            if (rolloffEl) rolloffEl.textContent = "3,820 Hz (Cutoff)";
            if (flatnessEl) flatnessEl.textContent = "0.0480 (Buzzy)";
            if (insightBox) insightBox.innerHTML = `<strong>Vocoder Artifact:</strong> Telltale steep energy drop-off at 3.8 kHz, unnatural rigid pitch contour, and missing high-frequency breath formants.`;
        });
    }

    start() {
        const render = () => {
            this.draw();
            this.animationId = requestAnimationFrame(render);
        };
        render();
    }

    draw() {
        const { ctx, canvas, currentMode } = this;
        const width = canvas.width;
        const height = canvas.height;

        ctx.clearRect(0, 0, width, height);

        this.phase += 0.04;

        // Draw Spectrum Bars
        const barWidth = width / this.numBars;
        const isHuman = currentMode === "human";

        for (let i = 0; i < this.numBars; i++) {
            const freqNorm = i / this.numBars; // 0 to 1 (0 to 8kHz)
            let amp = 0;

            if (isHuman) {
                // Human formants: F1 around 0.1, F2 around 0.25, F3 around 0.45, with natural vibrato
                const f1 = Math.exp(-Math.pow((freqNorm - 0.12), 2) / 0.008) * 0.9;
                const f2 = Math.exp(-Math.pow((freqNorm - 0.28), 2) / 0.012) * 0.7;
                const f3 = Math.exp(-Math.pow((freqNorm - 0.48), 2) / 0.02) * 0.5;
                const highNoise = Math.exp(-freqNorm * 2) * 0.25;
                const vibrato = 0.15 * Math.sin(this.phase * 2 + i * 0.3);

                amp = Math.max(0.05, f1 + f2 + f3 + highNoise + vibrato);
            } else {
                // Synthetic Vocoder: Rigid cutoff above 3.8 kHz (around freqNorm = 0.48), unnatural spikes
                if (freqNorm > 0.48) {
                    // Sharp brickwall cutoff!
                    amp = 0.02 + 0.015 * Math.random();
                } else {
                    const harmonicSpike = Math.pow(Math.sin(i * 1.4), 4) * 0.85;
                    const buzz = 0.1 * Math.sin(this.phase * 5 + i);
                    amp = Math.max(0.05, harmonicSpike + buzz);
                }
            }

            amp = Math.min(1.0, Math.max(0.04, amp));
            const barHeight = amp * (height * 0.65);
            const x = i * barWidth;
            const y = height - barHeight;

            // Bar Gradient
            const grad = ctx.createLinearGradient(0, y, 0, height);
            if (isHuman) {
                grad.addColorStop(0, "rgba(0, 255, 163, 0.9)");
                grad.addColorStop(1, "rgba(0, 229, 255, 0.15)");
            } else {
                if (freqNorm > 0.48) {
                    grad.addColorStop(0, "rgba(100, 116, 139, 0.3)");
                    grad.addColorStop(1, "rgba(30, 41, 59, 0.1)");
                } else {
                    grad.addColorStop(0, "rgba(239, 68, 68, 0.9)");
                    grad.addColorStop(1, "rgba(168, 85, 247, 0.2)");
                }
            }

            ctx.fillStyle = grad;
            ctx.fillRect(x + 1, y, barWidth - 2, barHeight);
        }

        // Draw Oscilloscope Overlay Waveform
        ctx.beginPath();
        ctx.lineWidth = 2;
        ctx.strokeStyle = isHuman ? "#00E5FF" : "#EF4444";

        const midY = height * 0.35;
        for (let x = 0; x < width; x += 3) {
            const t = (x / width) * 12 + this.phase;
            let waveVal = 0;

            if (isHuman) {
                // Smooth combined harmonics
                waveVal = Math.sin(t) * 25 + Math.sin(t * 2.3) * 12 + Math.sin(t * 4.7) * 6;
            } else {
                // Clipped / robotic square-ish wave
                waveVal = (Math.sin(t * 1.5) > 0 ? 1 : -1) * 22 + Math.sin(t * 5.5) * 8;
            }

            const y = midY + waveVal;
            if (x === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        }
        ctx.stroke();

        // Vocoder Cutoff Marker Line in Synth mode
        if (!isHuman) {
            const cutoffX = width * 0.48;
            ctx.save();
            ctx.setLineDash([4, 4]);
            ctx.strokeStyle = "rgba(239, 68, 68, 0.75)";
            ctx.lineWidth = 1.5;
            ctx.beginPath();
            ctx.moveTo(cutoffX, 0);
            ctx.lineTo(cutoffX, height);
            ctx.stroke();

            ctx.fillStyle = "#EF4444";
            ctx.font = "10px JetBrains Mono, monospace";
            ctx.fillText("3.8 kHz VOCODER CUTOFF", cutoffX + 6, 24);
            ctx.restore();
        }
    }
}

// ==========================================================================
// 5. NAVBAR SCROLL & MOBILE TOGGLE
// ==========================================================================
function setupNavigation() {
    const navbar = document.getElementById("navbar");
    const toggle = document.getElementById("mobileToggle");

    // Scroll styling
    window.addEventListener("scroll", () => {
        if (window.scrollY > 40) {
            navbar.classList.add("scrolled");
        } else {
            navbar.classList.remove("scrolled");
        }
    });

    // Mobile menu toggle
    toggle?.addEventListener("click", () => {
        navbar.classList.toggle("menu-open");
    });

    // Close mobile menu on click
    document.querySelectorAll(".nav-link").forEach(link => {
        link.addEventListener("click", () => {
            navbar.classList.remove("menu-open");
        });
    });
}

// ==========================================================================
// 6. INITIALIZATION
// ==========================================================================
document.addEventListener("DOMContentLoaded", () => {
    updateAllLaunchButtons();
    setupModal();
    setupNavigation();
    new AudioVisualizerSimulator("waveformCanvas");
    console.log("[VoxShield AI] Cyber-forensic landing page initialized.");
});
