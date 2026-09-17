/** Layout shell. The map fills the viewport; everything else floats above it. */

import { useEffect } from "react";
import { Moon, Sun } from "lucide-react";
import { MapView } from "./map/MapView";
import { AoiLayer } from "./map/AoiLayer";
import { ImageCanvas } from "./map/ImageCanvas";
import { CompassControl, ZoomControls } from "./map/MapControls";
import { Sidebar } from "./sidebar/Sidebar";
import { AICommandBar } from "./command/AICommandBar";
import { ResultOverlay } from "./results/ResultOverlay";
import { api } from "./state/api";
import { applyTheme, useAppStore } from "./state/useAppStore";
import { IconButton, Surface, ToastHost } from "./ui/primitives";

export default function App() {
  const theme = useAppStore((s) => s.theme);
  const toggleTheme = useAppStore((s) => s.toggleTheme);
  const modelIsFake = useAppStore((s) => s.modelIsFake);
  const setModelIsFake = useAppStore((s) => s.setModelIsFake);
  const setError = useAppStore((s) => s.setError);

  useEffect(() => applyTheme(theme), [theme]);

  useEffect(() => {
    api
      .health()
      .then((health) => setModelIsFake(health.model_is_fake))
      .catch(() => setError("Cannot reach the analysis server. Start it with: uvicorn satquery.server:app"));
  }, [setModelIsFake, setError]);

  return (
    <ToastHost>
      <main className="relative h-full w-full overflow-hidden">
        <MapView>
          <AoiLayer />
          <Sidebar />
          <ImageCanvas />
          <ResultOverlay />
          <AICommandBar />

          <div className="absolute top-4 right-4 z-20 flex items-center gap-2">
            {/* A stand-in model must be visible at all times, never mistaken for the real one. */}
            {modelIsFake && (
              <Surface className="px-2.5 py-1.5 text-[11px] font-medium text-warn">
                Demo model - answers are placeholders
              </Surface>
            )}
            <Surface className="p-1">
              <IconButton
                label={theme === "dark" ? "Switch to light map" : "Switch to dark map"}
                side="left"
                onClick={toggleTheme}
              >
                {theme === "dark" ? <Sun className="h-[18px] w-[18px]" strokeWidth={1.75} /> : <Moon className="h-[18px] w-[18px]" strokeWidth={1.75} />}
              </IconButton>
            </Surface>
            <CompassControl />
          </div>

          <div className="absolute right-4 bottom-28 z-20">
            <ZoomControls />
          </div>
        </MapView>
      </main>
    </ToastHost>
  );
}
