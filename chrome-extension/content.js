// test if it's actually sending to browser properly
console.log("Distress extension loaded on YouTube");

const IFRAME_ID = "distress-breathing-iframe";
let pollHandle = null;

function ensureOverlayIframe() {
    let iframe = document.getElementById(IFRAME_ID);
    if (iframe) return iframe;

    iframe = document.createElement("iframe");
    iframe.id = IFRAME_ID;
    iframe.src = chrome.runtime.getURL("breathing_exercise.html");
    iframe.style.position = "fixed";
    iframe.style.top = "0";
    iframe.style.left = "0";
    iframe.style.width = "100vw";
    iframe.style.height = "100vh";
    iframe.style.border = "0";
    iframe.style.zIndex = "2147483647";
    iframe.style.display = "none";

    document.documentElement.appendChild(iframe);
    return iframe;
}

function showOverlay() {
    const iframe = ensureOverlayIframe();
    if (iframe.style.display !== "block") {
        iframe.style.display = "block";
    }
}

function hideOverlay() {
    const iframe = document.getElementById(IFRAME_ID);
    if (iframe && iframe.style.display !== "none") {
        iframe.style.display = "none";
    }
}

function pauseYouTube() {
    const video = document.querySelector("video");
    if (video && !video.paused) video.pause();
}

function stopPolling() {
    if (pollHandle) clearInterval(pollHandle);
    pollHandle = null;
    console.log("Polling stopped due to extension context invalidation.");
}

async function getStateSafe() {
    try {
        return await new Promise((resolve, reject) => {
            try {
                chrome.runtime.sendMessage({ type: "GET_STATE" }, (response) => {
                    if (chrome.runtime.lastError) {
                        reject(chrome.runtime.lastError);
                    } else {
                        resolve(response);
                    }
                });
            } catch (e) {
                reject(e);
            }
        });
    } catch (e) {
        const msg = String(e && e.message ? e.message : e);
        if (msg.includes("Extension context invalidated")) {
            stopPolling();
            return null;
        }
        // Other errors (e.g. background not ready) - just ignore for this tick
        return null;
    }
}

async function pollStateAndRender() {
    const resp = await getStateSafe();

    if (!resp?.ok) {
        return;
    }

    const state = resp.data;
    const shouldShow = (state?.state === "INTERVENTION");

    if (shouldShow) {
        showOverlay();
        pauseYouTube();
    } else {
        hideOverlay();
    }
}

function startPolling() {
    if (pollHandle) return;

    // Initial Ensure
    ensureOverlayIframe();

    pollHandle = setInterval(pollStateAndRender, 500);
    pollStateAndRender();
}

startPolling();
