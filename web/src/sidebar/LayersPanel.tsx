/** Images and imagery settings.
 *
 *  Date and modality are real: they become `ImageInput.acquired` and `ImageInput.modality`, which
 *  decide how the agent routes the request. The catalogue-search parameters below them are not
 *  connected to anything and say so, because no imagery catalogue exists in this build. */

import { ChevronDown, ChevronUp, Eye, EyeOff, ImageOff, Layers, Trash2 } from "lucide-react";
import { useAppStore, selectAnalysisImages, type Layer } from "../state/useAppStore";
import { useMap, fitBounds } from "../map/MapView";
import { Button, cx, inputClass, NotConnected, Segmented, SectionGroup, Tooltip } from "../ui/primitives";
import { EmptyState } from "./Sidebar";
import type { Modality } from "../state/types";

export function LayersPanel() {
  const layers = useAppStore((s) => s.layers);
  const chosen = selectAnalysisImages(layers);
  const chosenIds = new Set(chosen.map((l) => l.id));

  if (!layers.length) {
    return (
      <EmptyState icon={Layers}>
        No images yet. Use the <strong className="font-medium text-ink">+</strong> button in the command bar to add a
        GeoTIFF, or open Help to try a demo scenario.
      </EmptyState>
    );
  }

  return (
    <div>
      <ul className="border-b border-line">
        {layers.map((layer, index) => (
          <LayerRow
            key={layer.id}
            layer={layer}
            index={index}
            total={layers.length}
            usedForAnalysis={chosenIds.has(layer.id)}
          />
        ))}
      </ul>

      {layers.length > 2 && (
        <p className="border-b border-line px-4 py-2.5 text-[11px] leading-relaxed text-faint">
          The agent takes up to two images. The two marked <em>In use</em> are the ones your next question will run on.
        </p>
      )}

      <SectionGroup
        title="Imagery search"
        note="Not connected in this build - there is no imagery catalogue behind these."
      >
        <div className="space-y-3 opacity-60">
          <DisabledField label="Satellite / source" value="Any source" />
          <DisabledField label="Resolution" value="Any resolution" />
          <DisabledField label="Maximum cloud cover" value="Any" />
          <p className="text-[11px] leading-relaxed text-faint">
            Searching a catalogue by area and date would need a live imagery service. This build analyses the files you
            provide. Dates and optical/SAR on each image above are real and do affect the analysis.
          </p>
        </div>
      </SectionGroup>
    </div>
  );
}

function DisabledField({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <span className="mb-1 flex items-center gap-1.5">
        <span className="text-[11px] font-medium tracking-wide text-muted uppercase">{label}</span>
        <NotConnected>off</NotConnected>
      </span>
      <select disabled aria-label={`${label} (not connected)`} className={inputClass}>
        <option>{value}</option>
      </select>
    </div>
  );
}

function LayerRow({
  layer,
  index,
  total,
  usedForAnalysis,
}: {
  layer: Layer;
  index: number;
  total: number;
  usedForAnalysis: boolean;
}) {
  const updateLayer = useAppStore((s) => s.updateLayer);
  const removeLayer = useAppStore((s) => s.removeLayer);
  const reorderLayer = useAppStore((s) => s.reorderLayer);
  const map = useMap();

  const bounds = layer.summary.bounds_wgs84;

  return (
    <li className="border-b border-line px-4 py-3 last:border-b-0">
      <div className="flex items-start gap-2.5">
        <img
          src={layer.preview_url}
          alt=""
          className="h-10 w-10 shrink-0 rounded-md border border-line object-cover"
        />
        <div className="min-w-0 flex-1">
          <p className="truncate text-[12px] font-medium text-ink" title={layer.name}>
            {layer.name}
          </p>
          <p className="mt-0.5 text-[11px] text-faint">
            {layer.summary.width} x {layer.summary.height} px
            {layer.summary.crs ? ` - ${layer.summary.crs}` : " - no CRS"}
            {layer.summary.decimation > 1 && ` - read at 1/${layer.summary.decimation.toFixed(1)}`}
          </p>
          {usedForAnalysis && (
            <span className="mt-1 inline-flex rounded-full bg-accent-soft px-1.5 py-0.5 text-[10px] font-semibold tracking-wide text-accent uppercase">
              In use
            </span>
          )}
        </div>

        <div className="flex shrink-0 items-center">
          <Button
            size="sm"
            aria-label={layer.visible ? `Hide ${layer.name}` : `Show ${layer.name}`}
            disabled={!layer.mappable}
            onClick={() => updateLayer(layer.id, { visible: !layer.visible })}
            className="w-7 px-0 text-muted"
          >
            {layer.visible ? <Eye className="h-3.5 w-3.5" /> : <EyeOff className="h-3.5 w-3.5" />}
          </Button>
          <Button
            size="sm"
            aria-label={`Remove ${layer.name}`}
            onClick={() => removeLayer(layer.id)}
            className="w-7 px-0 text-muted"
          >
            <Trash2 className="h-3.5 w-3.5" />
          </Button>
        </div>
      </div>

      {!layer.mappable && (
        <p className="mt-2 flex items-start gap-1.5 rounded-md bg-sunken px-2 py-1.5 text-[11px] leading-relaxed text-muted">
          <ImageOff className="mt-px h-3.5 w-3.5 shrink-0 text-faint" strokeWidth={1.75} />
          No georeferencing, so it cannot be placed on the map. It still analyses normally.
        </p>
      )}
      {layer.summary.georeference_note && (
        <p className="mt-2 rounded-md bg-sunken px-2 py-1.5 text-[11px] leading-relaxed text-muted">
          {layer.summary.georeference_note}.
        </p>
      )}

      {/* Real parameters: both feed straight into the analysis request. */}
      <div className="mt-2.5 flex flex-wrap items-center gap-2">
        <Segmented<Modality>
          value={layer.modality}
          onChange={(modality) => updateLayer(layer.id, { modality })}
          options={[
            { value: "optical", label: "Optical" },
            { value: "sar", label: "SAR" },
          ]}
        />
        <Tooltip label="Acquisition date: orders a before/after pair" side="top">
          <input
            type="date"
            value={layer.acquired ?? ""}
            aria-label={`Acquisition date for ${layer.name}`}
            onChange={(event) => updateLayer(layer.id, { acquired: event.target.value || null })}
            className={cx(inputClass, "w-[132px] py-1")}
          />
        </Tooltip>
      </div>

      {layer.mappable && (
        <div className="mt-2.5 flex items-center gap-2">
          <label className="flex flex-1 items-center gap-2">
            <span className="text-[11px] text-faint">Opacity</span>
            <input
              type="range"
              min={0.1}
              max={1}
              step={0.05}
              value={layer.opacity}
              disabled={!layer.visible}
              aria-label={`Opacity of ${layer.name}`}
              onChange={(event) => updateLayer(layer.id, { opacity: Number(event.target.value) })}
              className="h-1 flex-1 accent-[rgb(var(--accent))]"
            />
          </label>
          <Button
            size="sm"
            className="text-muted"
            disabled={!bounds}
            onClick={() => map && bounds && fitBounds(map, bounds)}
          >
            Zoom
          </Button>
          <div className="flex">
            <Button
              size="sm"
              aria-label={`Move ${layer.name} down`}
              disabled={index === 0}
              onClick={() => reorderLayer(layer.id, -1)}
              className="w-6 px-0 text-muted"
            >
              <ChevronDown className="h-3.5 w-3.5" />
            </Button>
            <Button
              size="sm"
              aria-label={`Move ${layer.name} up`}
              disabled={index === total - 1}
              onClick={() => reorderLayer(layer.id, 1)}
              className="w-6 px-0 text-muted"
            >
              <ChevronUp className="h-3.5 w-3.5" />
            </Button>
          </div>
        </div>
      )}
    </li>
  );
}
