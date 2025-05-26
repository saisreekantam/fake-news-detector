chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.type === "EXTRACT_INFO") {
    console.log("I recieved a request for scraping...");
    const getMeta = (name) =>
      document.querySelector(`meta[name='${name}']`)?.content ||
      document.querySelector(`meta[property='${name}']`)?.content ||
      null;

    // Try getting the URL from canonical or fallback to location
    const getCanonicalUrl = () =>
      document.querySelector("link[rel='canonical']")?.href || location.href;

    // Extract images: filter out blank or small ones
      let image = document.querySelector("meta[property='og:image']")?.content;

      if (!image) {
        const highPriorityImage = Array.from(document.images)
          .find(img => img.getAttribute('fetchpriority') === 'high');
        image = highPriorityImage?.src || null;
      }


    const responseData = {
      title: getMeta("og:title") || document.title || "No title",
      description: getMeta("description") || getMeta("og:description") || "No description",
      content: document.body.innerText.slice(0, 1000), // First 1000 chars as preview content
      url: getCanonicalUrl(),
      image: image,
      published_at: getMeta("article:published_time") || null,
      source_name: getMeta("og:site_name") || location.hostname
    };
    console.log("sending data....");
    console.log(responseData);
    sendResponse(responseData);
  }
});
