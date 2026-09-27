/** Where a weather answer came from. Every value is read from the server's response.
 *
 *  A forecast is for one point, and a drawn area can be large, so the card names the point (also
 *  marked on the map) and the area it stands for. The attribution is required by the data licence
 *  (CC BY 4.0) and is always shown. Weather is an optional capability, not an SIH requirement. */

import { useAppStore } from "../state/useAppStore";

function Field({ label, value, title }: { label: string; value: string; title?: string }) {
  return (
    <div className="min-w-0">
      <dt className="text-[10px] font-medium tracking-wide text-faint uppercase">{label}</dt>
      <dd className="truncate text-[12px] text-ink" title={title ?? value}>
        {value}
      </dd>
    </div>
  );
}

/** "19.075°N, 72.995°E" from (longitude, latitude). */
export function formatPoint([longitude, latitude]: [number, number]): string {
  return `${Math.abs(latitude).toFixed(3)}°${latitude >= 0 ? "N" : "S"}, ${Math.abs(longitude).toFixed(3)}°${
    longitude >= 0 ? "E" : "W"
  }`;
}

export function WeatherProvenance() {
  const weather = useAppStore((s) => s.result?.weather);
  if (!weather) return null;

  const [width, height] = weather.area_extent_km;
  const size = `~${width.toFixed(width < 1 ? 1 : 0)} × ${height.toFixed(height < 1 ? 1 : 0)} km`;
  const retrieved = weather.retrieved_at ? `${weather.retrieved_at.slice(0, 16).replace("T", " ")} UTC` : "not retrieved";

  return (
    <section className="rounded-[var(--radius-md)] border border-line bg-surface px-3 py-2.5">
      <dl className="grid grid-cols-2 gap-x-3 gap-y-2">
        <Field label="Forecast point" value={formatPoint(weather.point_wgs84)} title="Marked on the map" />
        <Field label="Stands for" value={`${weather.area_source}, ${size}`} />
        <Field
          label="Period"
          value={weather.period ? `${weather.period[0]} to ${weather.period[1]}` : "none"}
          title={weather.timezone ? `Local dates in ${weather.timezone}` : undefined}
        />
        <Field label="Retrieved" value={`${retrieved}${weather.cached ? " (cached)" : ""}`} />
      </dl>
      <p className="mt-2 border-t border-line pt-2 text-[10px] leading-relaxed text-faint">
        {weather.provider} forecast
        {weather.grid_point_wgs84 && ` · model grid point ${formatPoint(weather.grid_point_wgs84)}`}
        {weather.elevation_m !== null && ` · ${weather.elevation_m.toFixed(0)} m`}
        {weather.timezone && ` · ${weather.timezone}`}
        <br />
        <a href={weather.attribution_url} target="_blank" rel="noopener" className="underline underline-offset-2">
          {weather.attribution}
        </a>
        {" · "}Weather is an optional SatQuery capability, separate from the satellite analysis.
      </p>
    </section>
  );
}
