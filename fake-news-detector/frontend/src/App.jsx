import React, { useState,useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { BarChart2, Info, AlertCircle } from "lucide-react";
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

    const CircularProgress = ({ value = 0, color = "#4ade80", size = 80, label = "", subtitle = "" }) => {
      const strokeWidth = 8;
      const radius = (size - strokeWidth) / 2;
      const circumference = 2 * Math.PI * radius;
      const strokeDashoffset = circumference - (value / 100) * circumference;

      return (
        <div className="flex flex-col items-center w-[120px]">
          <div className="relative" style={{ width: size, height: '100px' }}>
            <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
              {/* Background Circle */}
              <circle
                cx={size / 2}
                cy={size / 2}
                r={radius}
                fill="none"
                stroke="#374151"
                strokeWidth={strokeWidth}
              />
              {/* Progress Circle */}
              <motion.circle
                cx={size / 2}
                cy={size / 2}
                r={radius}
                fill="none"
                stroke={color}
                strokeWidth={strokeWidth}
                strokeDasharray={circumference}
                initial={{ strokeDashoffset: circumference }}
                animate={{ strokeDashoffset }}
                transition={{ duration: 1.5, ease: "easeOut" }}
                strokeLinecap="round"
                transform={`rotate(-90 ${size / 2} ${size / 2})`}
              />
            </svg>

            {/* Center Value Text */}
            <div className="absolute inset-0 flex flex-col items-center justify-center">
              <motion.span
                initial={{ opacity: 0, scale: 0.8 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: 0.5 }}
                className="text-base font-bold text-white"
              >
                {Math.round(value)}
              </motion.span>
              <motion.span
                initial={{ opacity: 0 }}
                animate={{ opacity: 0.7 }}
                transition={{ delay: 0.7 }}
                className="text-[10px] text-gray-400"
              >
                /100
              </motion.span>
            </div>
          </div>

          {/* Label */}
          <span className="mt-2 text-xs font-semibold text-white text-center">{label}</span>

          {/* Subtitle */}
          {subtitle && (
            <motion.div
              initial={{ opacity: 0, y: 3 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.6 }}
              className="mt-1"
            >
              <span
                className="text-[10px] px-2 py-0.5 rounded-full"
                style={{
                  backgroundColor: `${color}30`,
                  color: color,
                }}
              >
                {subtitle}
              </span>
            </motion.div>
          )}
        </div>
      );
    };


  const getCredibilityColor = (score) => {
    if (score >= 80) return "#4ade80"; // Green
    if (score >= 60) return "#facc15"; // Yellow
    return "#f87171"; // Red
  };

  const getSentimentColor = (polarity) => {
    if (polarity >= 0.3) return "#4ade80"; // Green
    if (polarity >= -0.3) return "#facc15"; // Yellow
    return "#f87171"; // Red
  };

  const getExaggerationColor = (score) => {
    if (score <= 0.3) return "#4ade80"; // Green - Low exaggeration is good
    if (score <= 0.6) return "#facc15"; // Yellow
    return "#f87171"; // Red
  };

  // handleExtract();
  useEffect(() => {
    handleExtract();
  }, []);
  
  console.log("Rendering App...");
  return (
        <div className="flex flex-col h-[520px] w-[340px] font-sans bg-gradient-to-b from-purple-800 via-purple-900 to-indigo-950 text-white rounded-lg shadow-xl overflow-hidden">
          {/* Header */}
          <header className="border-b border-purple-600 px-4 py-3">
            <motion.div
              initial={{ opacity: 0, y: -10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.4 }}
              className="flex items-center justify-between"
            >
              <div className="flex items-center gap-2 justify-center w-full">
                <h1 className="text-lg font-bold text-purple-100">NewsChecker</h1>
              </div>
            </motion.div>
          </header>

          {/* Main Content */}
          <main className="flex-1 overflow-y-auto px-4 py-4 space-y-5">
            <AnimatePresence mode="wait">
              {loading ? (
                <motion.div
                  key="loading"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  className="h-full flex flex-col items-center justify-center space-y-4"
                >
                  <div className="relative w-16 h-16">
                    <motion.div
                      animate={{
                        rotate: 360,
                        borderRadius: ["50%", "50%", "50%", "50%"],
                      }}
                      transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
                      className="absolute inset-0 border-[3px] border-t-purple-300 border-r-purple-400 border-b-purple-500 border-l-transparent rounded-full"
                    />
                    <motion.div
                      animate={{
                        rotate: -180,
                        scale: [1, 0.8, 1],
                      }}
                      transition={{ duration: 2, repeat: Infinity, ease: "easeInOut" }}
                      className="absolute inset-3 border-[2px] border-t-transparent border-r-purple-300 border-b-purple-400 border-l-purple-200 rounded-full"
                    />
                  </div>
                  <motion.p
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    transition={{ delay: 0.5 }}
                    className="text-sm text-purple-200"
                  >
                    Analyzing article...
                  </motion.p>
                </motion.div>
              ) : (
                <motion.div
                  key="content"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ duration: 0.4 }}
                  className="space-y-6"
                >
                  {/* Analysis Results Card */}
                  <motion.div
                    initial={{ y: 10, opacity: 0 }}
                    animate={{ y: 0, opacity: 1 }}
                    transition={{ duration: 0.4 }}
                    className="bg-gradient-to-b from-purple-700 to-purple-900 rounded-lg p-4 border border-purple-600 shadow-md"
                  >
                    <motion.div
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      transition={{ delay: 0.1 }}
                      className="mb-4"
                    >
                      <h2 className="text-base font-medium flex items-center gap-2 text-purple-200">
                        <span className="w-1.5 h-4 rounded-full bg-purple-400 inline-block"></span>
                        Analysis Results
                      </h2>
                    </motion.div>

                    {/* Circular Indicators */}
                    <div className="flex justify-between gap-4 analysis-circles" style={{backgroundColor:'purple'}}>
                      <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.2 }} className="circle">
                        <CircularProgress
                          value={analysis?.credibility?.score || 0}
                          color={getCredibilityColor(analysis?.credibility?.score || 0)}
                          size={80}
                          label="Credibility"
                          subtitle={analysis?.credibility?.label || "Unknown"}
                        />
                      </motion.div>

                      <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.3 }} className="circle">
                        <CircularProgress
                          value={((analysis?.content_analysis?.sentiment?.polarity || 0) + 1) * 50}
                          color={getSentimentColor(analysis?.content_analysis?.sentiment?.polarity || 0)}
                          size={80}
                          label="Sentiment"
                          subtitle={analysis?.content_analysis?.sentiment?.label || "Unknown"}
                        />
                      </motion.div>

                      <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.4 }} className="circle">
                        <CircularProgress
                          value={(analysis?.content_analysis?.bias_markers?.exaggeration || 0) * 100}
                          color={getExaggerationColor(analysis?.content_analysis?.bias_markers?.exaggeration || 0)}
                          size={80}
                          label="Exaggeration"
                          subtitle={
                            (analysis?.content_analysis?.bias_markers?.exaggeration || 0) < 0.3
                              ? "Low"
                              : (analysis?.content_analysis?.bias_markers?.exaggeration || 0) < 0.6
                              ? "Medium"
                              : "High"
                          }
                        />
                      </motion.div>
                    </div>
                  </motion.div>

                  {/* Explanation Section */}
                  <motion.div
                    initial={{ y: 10, opacity: 0 }}
                    animate={{ y: 0, opacity: 1 }}
                    transition={{ delay: 0.6, duration: 0.4 }}
                    className="bg-purple-950 rounded-xl border border-purple-700 shadow-lg overflow-hidden"
                  >
                    <div className="flex items-center gap-2 px-5 py-3 border-b border-purple-700 bg-purple-900/60">
                      <Info size={16} className="text-purple-400" />
                      <h3 className="text-sm font-semibold text-white tracking-wide">Analysis Explanation</h3>
                    </div>

                    <div className="relative px-5 py-4 bg-purple-950">
                      <motion.div
                        initial={{ width: "0%" }}
                        animate={{ width: "100%" }}
                        transition={{ delay: 0.8, duration: 0.8 }}
                        className="absolute left-0 top-0 h-full w-full bg-purple-500/5 pointer-events-none"
                      />
                      <p className="text-sm text-purple-200 leading-relaxed relative z-10">
                        {analysis?.credibility?.explanation || "No explanation available."}
                      </p>
                    </div>
                  </motion.div>
                </motion.div>
              )}
            </AnimatePresence>
          </main>

          {/* Footer */}
          {/* <footer className="border-t border-purple-700 py-2 px-4 flex justify-center items-center bg-gradient-to-r from-purple-800 via-purple-900 to-indigo-900">
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.9 }}
              className="text-xs text-purple-300"
            >
              Powered by NewsChecker AI
            </motion.div>
          </footer> */}
        </div>


  );  
}
  

export default App;
