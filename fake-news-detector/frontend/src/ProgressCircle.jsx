// ProgressCircle.js
import React from "react";
import "./ProgressCircle.css";

const ProgressCircle = ({ label, value, max = 100, color = "#00b894" }) => {
  const radius = 40;
  const stroke = 8;
  const normalizedRadius = radius - stroke * 0.5;
  const circumference = normalizedRadius * 2 * Math.PI;
  const percent = Math.min((value / max) * 100, 100);
  const strokeDashoffset = circumference - (percent / 100) * circumference;

  return (
    <div className="circle-container">
      <svg height={radius * 2} width={radius * 2}>
        <circle
          stroke="#ecf0f1"
          fill="transparent"
          strokeWidth={stroke}
          r={normalizedRadius}
          cx={radius}
          cy={radius}
        />
        <circle
          stroke={color}
          fill="transparent"
          strokeWidth={stroke}
          strokeLinecap="round"
          strokeDasharray={circumference + " " + circumference}
          style={{ strokeDashoffset, transition: "stroke-dashoffset 1s ease" }}
          r={normalizedRadius}
          cx={radius}
          cy={radius}
        />
        <text
          x="50%"
          y="50%"
          textAnchor="middle"
          dy=".3em"
          className="circle-value"
        >
          {value?.toFixed(0) ?? "0"}%
        </text>
      </svg>
      <div className="circle-label">{label}</div>
    </div>
  );
};

export default ProgressCircle;
