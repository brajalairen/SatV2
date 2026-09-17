/** Place search. This moves the camera and nothing else: it tells you where a place is, and makes
 *  no claim that imagery is available there. Imagery still comes from the files you add. */

import { useEffect, useRef, useState } from "react";
import { Search } from "lucide-react";
import { searchPlaces, type Place } from "../state/api";
import { useMap, fitBounds } from "../map/MapView";
import { cx, inputClass, Spinner } from "../ui/primitives";

export function SearchPanel() {
  const map = useMap();
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<Place[]>([]);
  const [status, setStatus] = useState<"idle" | "searching" | "empty" | "failed">("idle");
  const input = useRef<HTMLInputElement>(null);

  useEffect(() => input.current?.focus(), []);

  // Debounced so typing does not hammer the geocoder, and superseded requests are aborted.
  useEffect(() => {
    const term = query.trim();
    if (term.length < 3) {
      setResults([]);
      setStatus("idle");
      return;
    }
    const controller = new AbortController();
    const timer = setTimeout(async () => {
      setStatus("searching");
      try {
        const places = await searchPlaces(term, controller.signal);
        setResults(places);
        setStatus(places.length ? "idle" : "empty");
      } catch (error) {
        if ((error as Error).name === "AbortError") return;
        setStatus("failed");
      }
    }, 350);
    return () => {
      controller.abort();
      clearTimeout(timer);
    };
  }, [query]);

  const goTo = (place: Place) => {
    if (!map) return;
    if (place.bbox) fitBounds(map, place.bbox, 60);
    else map.flyTo({ center: [place.lon, place.lat], zoom: 11, duration: 800 });
  };

  return (
    <div className="p-4">
      <div className="relative">
        <Search className="pointer-events-none absolute top-1/2 left-2.5 h-3.5 w-3.5 -translate-y-1/2 text-faint" />
        <input
          ref={input}
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          placeholder="Town, city or region"
          aria-label="Search for a place"
          className={cx(inputClass, "pl-8")}
        />
        {status === "searching" && (
          <Spinner className="absolute top-1/2 right-2.5 -translate-y-1/2 text-faint" />
        )}
      </div>

      {status === "empty" && <p className="mt-3 text-[12px] text-muted">No places matched that.</p>}
      {status === "failed" && <p className="mt-3 text-[12px] text-danger">Place search is unavailable right now.</p>}

      {results.length > 0 && (
        <ul className="mt-3 -mx-1">
          {results.map((place) => (
            <li key={`${place.lon},${place.lat}`}>
              <button
                type="button"
                onClick={() => goTo(place)}
                className="w-full rounded-[var(--radius-sm)] px-3 py-2 text-left text-[12px] leading-snug text-ink transition-colors hover:bg-hover"
              >
                {place.name}
              </button>
            </li>
          ))}
        </ul>
      )}

      <p className="mt-4 border-t border-line pt-3 text-[11px] leading-relaxed text-faint">
        Search moves the map only. It does not fetch satellite imagery: add your own GeoTIFFs to analyse an area.
        Places from OpenStreetMap Nominatim.
      </p>
    </div>
  );
}
