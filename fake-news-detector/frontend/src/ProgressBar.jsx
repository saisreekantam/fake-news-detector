// ProgressBar.js
import React from "react";
import "./ProgressBar.css";

const ProgressBar = ({ label, value, max = 100, color = "#3498db" }) => {
  const percentage = Math.min((value / max) * 100, 100);

  return (
    <div className="progress-container">
      <div className="progress-label">
        <strong>{label}</strong> {value?.toFixed(2) ?? "N/A"} / {max}
      </div>
      <div className="progress-bar-bg">
        <div
          className="progress-bar-fill"
          style={{ width: `${percentage}%`, backgroundColor: color }}
        ></div>
      </div>
    </div>
  );
};

export default ProgressBar;
