const STATE_URL = "http://127.0.0.1:8765/state";

async function fetchState() {
    const res = await fetch(STATE_URL, { cache: "no-store" });
    if (!res.ok) throw new Error(`state fetch failed: ${res.status}`);
    return await res.json();
}

chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
    if (msg?.type !== "GET_STATE") return;

    (async () => {
        try {
            const data = await fetchState();
            sendResponse({ ok: true, data });
        } catch (e) {
            sendResponse({ ok: false, error: String(e?.message || e) });
        }
    })();

    return true;
});
