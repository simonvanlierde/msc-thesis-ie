import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { App } from "./App";
import "./index.css";

// maplibre-gl.css is imported by the two map components, so its 75 kB rides in their lazy
// chunk instead of blocking the first paint of a page that may never show a map.

const root = document.getElementById("root");
if (!root) throw new Error("#root not found");
createRoot(root).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
