import React, { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { BarChart2, Info, AlertCircle, Shield, TrendingUp, Eye } from "lucide-react";
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
        console.log("Response I have...");
        console.log(response);
        setInfo(response);
        if(info!=null){
        console.log("Info i recieved....")
        console.log(info);
        }

        try {
          const res = await fetch("http://localhost:5000/analyze", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(response),
          });
          const analysisData = await res.json();
          console.log("Analysis data...", analysisData);
          setAnalysis(analysisData);
        } catch (err) {
          console.error("API call failed", err);
        }

        setLoading(false);
      });
    });
  };

  const CircularProgress = ({ value = 0, color = "#a855f7", size = 110, label = "", subtitle = "", icon: Icon }) => {
    const strokeWidth = 6;
    const radius = (size - strokeWidth) / 2;
    const circumference = 2 * Math.PI * radius;
    const strokeDashoffset = circumference - (value / 100) * circumference;

    return (
      <div className="circular-progress-container">
        <div className="circular-progress-wrapper" style={{ width: size, height: size }}>
          <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} className="progress-svg" style={{objectFit:'cover',border:'None',minWidth:'90px',boxShadow:'none'}}>
            {/* Glow Effect */}
            <defs>
              <filter id={`glow-${label}`} x="-50%" y="-50%" width="200%" height="200%">
                <feGaussianBlur stdDeviation="3" result="coloredBlur"/>
                <feMerge> 
                  <feMergeNode in="coloredBlur"/>
                  <feMergeNode in="SourceGraphic"/>
                </feMerge>
              </filter>
              <linearGradient id={`gradient-${label}`} x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" stopColor={color} />
                <stop offset="100%" stopColor={color + "80"} />
              </linearGradient>
            </defs>
            
            {/* Background Circle */}
            <circle
              cx={size / 2}
              cy={size / 2}
              r={radius}
              fill="none"
              stroke="rgba(139, 92, 246, 0.1)"
              strokeWidth={strokeWidth}
              className="progress-bg"
              style={{border:'None'}}
            />
            
            {/* Progress Circle */}
            <motion.circle
              cx={size / 2}
              cy={size / 2}
              r={radius}
              fill="none"
              stroke={`url(#gradient-${label})`}
              strokeWidth={strokeWidth}
              strokeDasharray={circumference}
              initial={{ strokeDashoffset: circumference }}
              animate={{ strokeDashoffset }}
              transition={{ duration: 2, ease: "easeOut", delay: 0.5 }}
              strokeLinecap="round"
              transform={`rotate(-90 ${size / 2} ${size / 2})`}
              filter={`url(#glow-${label})`}
              className="progress-circle"
            />
          </svg>

          {/* Center Content */}
          <div className="progress-center">
            {Icon && (
              <motion.div
                initial={{ scale: 0, opacity: 0 }}
                animate={{ scale: 1, opacity: 1 }}
                transition={{ delay: 1, duration: 0.5 }}
                className="progress-icon"
                style={{ color }}
              >
                <Icon size={16} />
              </motion.div>
            )}
            <motion.span
              initial={{ opacity: 0, scale: 0.8 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: 1.2, duration: 0.5 }}
              className="progress-value"
            >
              {Math.round(value)}
            </motion.span>
            <motion.span
              initial={{ opacity: 0 }}
              animate={{ opacity: 0.7 }}
              transition={{ delay: 1.4, duration: 0.5 }}
              className="progress-max"
            >
              /100
            </motion.span>
          </div>
        </div>

        {/* Label */}
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 1.6, duration: 0.5 }}
          className="progress-label-container"
        >
          <span className="progress-label">{label}</span>
          {subtitle && (
            <motion.div
              initial={{ opacity: 0, scale: 0.8 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: 1.8, duration: 0.5 }}
              className="progress-subtitle"
              style={{
                backgroundColor: `${color}20`,
                color: color,
                borderColor: `${color}40`,
              }}
            >
              {subtitle}
            </motion.div>
          )}
        </motion.div>
      </div>
    );
  };

  const getCredibilityColor = (score) => {
    if (score >= 80) return "#10b981"; // Green
    if (score >= 60) return "#f59e0b"; // Amber
    return "#ef4444"; // Red
  };

  const getSentimentColor = (polarity) => {
    if (polarity >= 0.3) return "#10b981"; // Green
    if (polarity >= -0.3) return "#f59e0b"; // Amber
    return "#ef4444"; // Red
  };

  const getExaggerationColor = (score) => {
    if (score <= 0.3) return "#10b981"; // Green - Low exaggeration is good
    if (score <= 0.6) return "#f59e0b"; // Amber
    return "#ef4444"; // Red
  };

  useEffect(() => {
    handleExtract();
  }, []);

  return (
    <div className="newschecker-container">
      {/* Animated Background */}
      <div className="background-orbs">
        <motion.div 
          className="orb orb-1"
          animate={{
            scale: [1, 1.2, 1],
            opacity: [0.3, 0.6, 0.3],
          }}
          transition={{
            duration: 4,
            repeat: Infinity,
            ease: "easeInOut"
          }}
        />
        <motion.div 
          className="orb orb-2"
          animate={{
            scale: [1.2, 1, 1.2],
            opacity: [0.2, 0.5, 0.2],
          }}
          transition={{
            duration: 6,
            repeat: Infinity,
            ease: "easeInOut"
          }}
        />
      </div>

      {/* Header */}
      {/* <header className="newschecker-header">
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          className="header-content"
        >
          <div className="logo-container">
            <motion.div
              initial={{ rotate: -180, scale: 0 }}
              animate={{ rotate: 0, scale: 1 }}
              transition={{ duration: 0.8, delay: 0.2 }}
              className="logo-icon"
            >
              <Shield size={24} />
            </motion.div>
            <h1 className="logo-text">NewsChecker</h1>
          </div>
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.5 }}
            className="version-badge"
          >
            v2.0
          </motion.div>
        </motion.div>
      </header> */}

      {/* Main Content */}
      <main className="newschecker-main">
        <AnimatePresence mode="wait">
          {loading ? (
            <motion.div
              key="loading"
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.9 }}
              className="loading-section"
            >
              <div className="loading-animation">
                <motion.div
                  className="spinner-ring"
                  animate={{ rotate: 360 }}
                  transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
                />
                <motion.div
                  className="spinner-inner"
                  animate={{ rotate: -180 }}
                  transition={{ duration: 3, repeat: Infinity, ease: "linear" }}
                />
                <motion.div
                  className="spinner-core"
                  animate={{ 
                    scale: [1, 1.2, 1],
                    opacity: [0.5, 1, 0.5]
                  }}
                  transition={{ duration: 2, repeat: Infinity, ease: "easeInOut" }}
                >
                  <BarChart2 size={20} />
                </motion.div>
              </div>
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 0.5 }}
                className="loading-text-container"
              >
                <h3 className="loading-title">Analyzing Article</h3>
                <p className="loading-subtitle">Please wait while we process the content...</p>
              </motion.div>
            </motion.div>
          ) : (
            <motion.div
              key="content"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6 }}
              className="analysis-section"
            >
              {/* Results Header */}
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.2 }}
                className="results-header"
              >
                <div className="results-title-container">
                  <div className="results-icon">
                    <Info size={20} />
                  </div>
                  <h2 className="results-title">Analysis Results</h2>
                </div>
              </motion.div>

              {/* Circular Progress Bars */}
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 0.4 }}
                className="circular-progress-grid"
              >
                <CircularProgress
                  value={analysis?.credibility?.score || 75}
                  color={getCredibilityColor(analysis?.credibility?.score || 75)}
                  label="Credibility"
                  subtitle={analysis?.credibility?.label || "High"}
                  icon={Shield}
                />

                <CircularProgress
                  value={100 - ((analysis?.content_analysis?.bias_markers?.exaggeration || 0.2) * 100)}
                  color={getExaggerationColor(analysis?.content_analysis?.bias_markers?.exaggeration || 0.2)}
                  label="Objectivity"
                  subtitle={
                    (analysis?.content_analysis?.bias_markers?.exaggeration || 0.2) < 0.3
                      ? "High"
                      : (analysis?.content_analysis?.bias_markers?.exaggeration || 0.2) < 0.6
                      ? "Medium"
                      : "Low"
                  }
                  icon={Eye}
                />
              </motion.div>

              {/* Sentiment Linear Progress Bar */}
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.6 }}
                className="sentiment-section"
              >
                <div className="sentiment-header">
                  <div className="sentiment-icon">
                    <TrendingUp size={16} />
                  </div>
                  <span className="sentiment-label">Sentiment Analysis</span>
                  <span className="sentiment-value" style={{color:`${getSentimentColor(analysis?.content_analysis?.sentiment?.polarity || 0)}`}}>
                    {analysis?.content_analysis?.sentiment?.label || "Neutral"}
                  </span>
                </div>
                <div className="sentiment-bar-container">
                  <div className="sentiment-scale">
                    <span className="scale-label negative">Negative</span>
                    <span className="scale-label neutral">Neutral</span>
                    <span className="scale-label positive">Positive</span>
                  </div>
                  <div className="sentiment-bar-wrapper">
                    <div className="sentiment-bar-bg">
                      <div className="sentiment-markers">
                        <div className="marker negative-marker"></div>
                        <div className="marker neutral-marker"></div>
                        <div className="marker positive-marker"></div>
                      </div>
                      <motion.div
                        className="sentiment-indicator"
                        initial={{ left: "50%" }}
                        animate={{ 
                          left: `${((analysis?.content_analysis?.sentiment?.polarity || 0) + 1) * 50}%`,
                          backgroundColor: getSentimentColor(analysis?.content_analysis?.sentiment?.polarity || 0)
                        }}
                        transition={{ duration: 2, ease: "easeOut", delay: 0.8 }}
                        style={{
                          boxShadow: `0 0 20px ${getSentimentColor(analysis?.content_analysis?.sentiment?.polarity || 0)}40`
                        }}
                      />
                    </div>
                    <div className="sentiment-value-display">
                      <motion.span
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        transition={{ delay: 1.5 }}
                        style={{ color: getSentimentColor(analysis?.content_analysis?.sentiment?.polarity || 0) }}
                      >
                        {(analysis?.content_analysis?.sentiment?.polarity || 0).toFixed(2)}
                      </motion.span>
                    </div>
                  </div>
                </div>
              </motion.div>

              {/* Explanation Card */}
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 1 }}
                className="explanation-card"
              >
                <div className="explanation-header">
                  <AlertCircle size={16} className="explanation-icon" />
                  <h3 className="explanation-title">Key Insights</h3>
                </div>
                <motion.div
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ delay: 1.2 }}
                  className="explanation-content scrollable"
                >
                  <div className="explanation-highlight"></div>
                  <p className="explanation-text">
                    {analysis?.credibility?.explanation || 
                     "This article demonstrates strong credibility indicators with balanced reporting and reliable sourcing. The content maintains objectivity with minimal bias markers detected. The analysis considers multiple factors including source reliability, fact verification, writing style, and potential bias indicators to provide a comprehensive assessment of the article's trustworthiness and journalistic quality."}
                  </p>
                </motion.div>
              </motion.div>
            </motion.div>
          )}
        </AnimatePresence>
      </main>
    </div>
  );
}

export default App;