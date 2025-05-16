// background.js

// On extension installation
chrome.runtime.onInstalled.addListener(() => {
  console.log("✅ Fake News Detector Extension installed.");
});

// Listener for messages from popup or content script
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.type === "CHECK_NEWS") {
    console.log("🧠 Background received news to check:", message.text);

    // In the future, call ML API here and send response back
    const isFake = message.text.toLowerCase().includes("fake"); // dummy logic

    sendResponse({
      result: isFake ? "⚠️ This might be fake news!" : "✅ Looks real.",
    });

    return true; // Keep the message channel open for async response
  }
});
