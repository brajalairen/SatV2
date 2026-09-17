import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import "./map/worker"; // must run before any Map is created
import App from "./App";
import "./index.css";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
