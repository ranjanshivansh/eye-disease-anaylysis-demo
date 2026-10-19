import { useState } from "react";
import Analyze from "./pages/Analyze.jsx";
import History from "./pages/History.jsx";
import Compare from "./pages/Compare.jsx";
import ModelPage from "./pages/ModelPage.jsx";

const TABS = [
  ["analyze", "Analyze"],
  ["compare", "Compare models"],
  ["history", "History"],
  ["model", "Model & results"],
];

export default function App() {
  const [tab, setTab] = useState("analyze");
  const [last, setLast] = useState(null); // keeps the last analysis when switching tabs
  return (
    <>
      <header className="top">
        <div className="brand">
          <span className="dot" /> Retinal Screening Demo
        </div>
        <nav>
          {TABS.map(([k, label]) => (
            <button key={k} className={tab === k ? "on" : ""} onClick={() => setTab(k)}>
              {label}
            </button>
          ))}
        </nav>
      </header>
      <div className="banner">
        Research prototype. Not a medical device and not for diagnosis. Heatmaps are qualitative and need clinician review.
      </div>
      <main>
        {tab === "analyze" && <Analyze last={last} setLast={setLast} />}
        {tab === "compare" && <Compare />}
        {tab === "history" && <History />}
        {tab === "model" && <ModelPage />}
      </main>
    </>
  );
}
