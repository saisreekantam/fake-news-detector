chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.type === "EXTRACT_INFO") {
    const bodyText = document.body.innerText;

    const getMeta = (name) =>
      document.querySelector(`meta[name='${name}']`)?.content ||
      document.querySelector(`meta[property='${name}']`)?.content ||
      "Not available";

    // Extract images: filter out blank or small ones
    const images = Array.from(document.images)
      .filter(img => img.src && img.naturalWidth > 100 && img.naturalHeight > 100)
      .map(img => img.src);

    const metadata = {
      title: document.title || "No title",
      description: getMeta("description"),
      ogTitle: getMeta("og:title"),
      ogDesc: getMeta("og:description"),
      author: getMeta("author")
    };

    sendResponse({ 
      text: bodyText.slice(0, 1000), // limit for preview
      metadata,
      images
    });
  }
});
