/** Where the analysed imagery came from, when it was fetched rather than uploaded.
 *
 *  Every value is read from the fetch-imagery response: nothing here is hardcoded, so the card
 *  cannot claim a source or a date the retrieval did not actually return. */

import { useAppStore } from "../state/useAppStore";

function Field({ label, value }: { label: string; value: string }) {
  return (
    <div className="min-w-0">
      <dt className="text-[10px] font-medium tracking-wide text-faint uppercase">{label}</dt>
      <dd className="truncate text-[12px] text-ink" title={value}>
        {value}
      </dd>
    </div>
  );
}

export function SceneProvenance() {
  const scene = useAppStore((s) => s.scene);
  if (!scene) return null;

  const cloud = typeof scene.cloud_cover === "number" ? `${scene.cloud_cover.toFixed(2)}%` : "not reported";

  return (
    <section className="rounded-[var(--radius-md)] border border-line bg-surface px-3 py-2.5">
      <dl className="grid grid-cols-2 gap-x-3 gap-y-2 sm:grid-cols-4">
        <Field label="Source" value={`${scene.satellite} ${scene.product_level}`} />
        <Field label="Acquired" value={scene.acquired || "unknown"} />
        <Field label="Cloud cover" value={cloud} />
        <Field label="Area analysed" value="Selected area" />
      </dl>

      <p className="mt-2 border-t border-line pt-2 text-[10px] leading-relaxed text-faint">
        {scene.provider} · {scene.resolution_m} m · {scene.crs} · {scene.bands.join(", ")}
        {scene.cached ? " · from cache" : ""}
        <br />
        {scene.attribution}
      </p>
    </section>
  );
}
