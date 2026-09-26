/** Where the analysed imagery came from, when it was fetched rather than uploaded.
 *
 *  Every value is read from the fetch-imagery response: nothing here is hardcoded, so the card
 *  cannot claim a source or a date the retrieval did not actually return. A comparison shows both
 *  acquisitions, each with its own date, cloud cover and scene id, so it is plain that they are two
 *  real scenes and not one image shown twice. */

import { useAppStore } from "../state/useAppStore";
import type { SceneMetadata } from "../state/types";

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

const cloudText = (scene: SceneMetadata) =>
  typeof scene.cloud_cover === "number" ? `${scene.cloud_cover.toFixed(2)}%` : "not reported";

function daysBetween(earlier: string, later: string): number {
  return Math.round((Date.parse(later) - Date.parse(earlier)) / 86_400_000);
}

export function SceneProvenance() {
  const scenes = useAppStore((s) => s.scenes);
  if (!scenes.length) return null;
  const latest = scenes[scenes.length - 1]!;

  return (
    <section className="rounded-[var(--radius-md)] border border-line bg-surface px-3 py-2.5">
      {scenes.length === 2 ? (
        <Comparison before={scenes[0]!} after={scenes[1]!} />
      ) : (
        <dl className="grid grid-cols-2 gap-x-3 gap-y-2 sm:grid-cols-4">
          <Field label="Source" value={`${latest.satellite} ${latest.product_level}`} />
          <Field label="Acquired" value={latest.acquired || "unknown"} />
          <Field label="Cloud cover" value={cloudText(latest)} />
          <Field label="Area analysed" value="Selected area" />
        </dl>
      )}

      <p className="mt-2 border-t border-line pt-2 text-[10px] leading-relaxed text-faint">
        {latest.provider} · {latest.resolution_m} m · {latest.crs} · {latest.bands.join(", ")}
        {scenes.every((scene) => scene.cached) ? " · from cache" : ""}
        <br />
        {latest.attribution}
      </p>
    </section>
  );
}

function Comparison({ before, after }: { before: SceneMetadata; after: SceneMetadata }) {
  // One row per fact, both dates side by side: compact, so the card stays clear of the map controls.
  const source = (scene: SceneMetadata) => `${scene.satellite} ${scene.product_level}`;
  const rows: [string, string, string, string | undefined, string | undefined][] = [
    ["Acquired", before.acquired || "unknown", after.acquired || "unknown", before.scene_id ?? undefined, after.scene_id ?? undefined],
    ["Cloud cover", cloudText(before), cloudText(after), undefined, undefined],
  ];
  return (
    <>
      <dl className="grid grid-cols-[auto_1fr_1fr] items-baseline gap-x-3 gap-y-1">
        <span />
        <span className="text-[10px] font-semibold tracking-wide text-muted uppercase">Before</span>
        <span className="text-[10px] font-semibold tracking-wide text-muted uppercase">After</span>
        {rows.map(([label, first, second, firstTitle, secondTitle]) => (
          <div key={label} className="contents">
            <dt className="text-[10px] font-medium tracking-wide text-faint uppercase">{label}</dt>
            <dd className="truncate text-[12px] text-ink" title={firstTitle ?? first}>{first}</dd>
            <dd className="truncate text-[12px] text-ink" title={secondTitle ?? second}>{second}</dd>
          </div>
        ))}
      </dl>
      <p className="mt-1.5 text-[11px] text-muted">
        {source(before) === source(after) ? source(after) : `${source(before)} / ${source(after)}`} · two acquisitions,{" "}
        {daysBetween(before.acquired, after.acquired)} days apart · selected area
      </p>
    </>
  );
}
