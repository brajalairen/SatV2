/** MapLibre v6 loads its tile-parsing worker as a separate module. Vite must bundle and fingerprint
 *  that worker itself, otherwise the browser asks for an /assets/maplibre-gl-worker.mjs that was
 *  never emitted and tile parsing silently fails. Import this before the first Map is constructed. */

import { setWorkerUrl } from "maplibre-gl";
import workerUrl from "maplibre-gl/dist/maplibre-gl-worker.mjs?worker&url";

setWorkerUrl(workerUrl);
