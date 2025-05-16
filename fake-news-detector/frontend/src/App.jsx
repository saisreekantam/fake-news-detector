import React, { useState } from "react";
import "./App.css";

function App() {
  const [info, setInfo] = useState(null);
  const [analysis, setAnalysis] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleExtract = () => {
    setLoading(true);
    chrome.tabs.query({ active: true, currentWindow: true }, ([tab]) => {
      chrome.tabs.sendMessage(tab.id, { type: "EXTRACT_INFO" }, async (response) => {
        if (chrome.runtime.lastError || !response) {
          console.error("Content script error", chrome.runtime.lastError);
          setLoading(false);
          return;
        }

        setInfo(response);

        try {
          const res = await fetch("http://localhost:5000/analyze", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(response),
          });

          const analysisData = await res.json();
          setAnalysis(analysisData);
        } catch (err) {
          console.error("API call failed", err);
        }

        setLoading(false);
      });
    });
  };

  return (
    <div className="App" style={{ width: "400px", padding: "1rem" }}>
      <h2>🧠 News Checker</h2>
      <button onClick={handleExtract} disabled={loading}>
        {loading ? "Analyzing..." : "Analyze This Page"}
      </button>

      {info && (
        <div style={{ marginTop: "1rem" }}>
          <h3>🔎 Extracted Info:</h3>
          <p><strong>Title:</strong> {info.metadata.title}</p>
          <p><strong>Description:</strong> {info.metadata.description}</p>
          <p><strong>Author:</strong> {info.metadata.author}</p>

          <h4>📷 Images:</h4>
          {info.images && info.images.length > 0 ? (
            <div style={{ display: "flex", gap: "5px", flexWrap: "wrap" }}>
              {info.images.slice(0, 3).map((img, i) => (
                <img key={i} src={img} alt="img" width={80} />
              ))}
            </div>
          ) : <p>No images found.</p>}
        </div>
      )}

      {analysis && (
        <div style={{ marginTop: "1rem" }}>
          <h3>📊 Analysis:</h3>
          <p><strong>Credibility Score:</strong> {analysis.credibility_score} / 100</p>
          <p><strong>Sentiment:</strong> {analysis.sentiment}</p>
          <p><strong>Bias Markers:</strong> {analysis.bias_markers.join(', ')}</p>
          <p><strong>Explanation:</strong> {analysis.explanation}</p>

          <h4>🔗 Fact-Check Links:</h4>
          <ul>
            {analysis.fact_check_links.map((link, i) => (
              <li key={i}>
                <a href={link} target="_blank" rel="noreferrer">{link}</a>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

export default App;
