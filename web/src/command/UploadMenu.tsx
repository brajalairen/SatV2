/** The + menu. GeoTIFF and plain images are real; PDF and other data are shown disabled, because
 *  the analysis pipeline accepts rasters only and pretending otherwise would be a lie. */

import { useRef, useState } from "react";
import { FileImage, FileText, Image, Layers3, Database } from "lucide-react";
import { api, ApiError } from "../state/api";
import { useAppStore } from "../state/useAppStore";
import { useMap, fitBounds } from "../map/MapView";
import { MenuItem, Popover, Spinner, useToast } from "../ui/primitives";

const GEOTIFF = ".tif,.tiff";
const PLAIN = ".png,.jpg,.jpeg";

export function UploadMenu({ open, onClose }: { open: boolean; onClose: () => void }) {
  const input = useRef<HTMLInputElement>(null);
  const accept = useRef<string>(GEOTIFF);
  const [busy, setBusy] = useState(false);

  const addLayers = useAppStore((s) => s.addLayers);
  const setError = useAppStore((s) => s.setError);
  const map = useMap();
  const toast = useToast();

  const pick = (types: string) => {
    accept.current = types;
    onClose();
    // The picker has to open from the click, so this runs synchronously.
    if (input.current) {
      input.current.accept = types;
      input.current.click();
    }
  };

  const onFiles = async (files: FileList | null) => {
    if (!files?.length) return;
    setBusy(true);
    setError(null);
    const uploaded = [];
    for (const file of Array.from(files).slice(0, 2)) {
      try {
        uploaded.push(await api.upload(file, "optical"));
      } catch (error) {
        setError(error instanceof ApiError ? error.message : `Could not read ${file.name}.`);
      }
    }
    setBusy(false);
    if (!uploaded.length) return;

    addLayers(uploaded);
    const bounds = uploaded.find((u) => u.summary.bounds_wgs84)?.summary.bounds_wgs84;
    if (map && bounds) fitBounds(map, bounds);
    const offMap = uploaded.filter((u) => !u.mappable);
    if (offMap.length) toast(`${offMap[0]!.name} has no georeferencing, so it opens in the image viewer.`);
    if (input.current) input.current.value = "";
  };

  return (
    <>
      <input
        ref={input}
        type="file"
        multiple
        accept={accept.current}
        className="sr-only"
        onChange={(event) => onFiles(event.target.files)}
      />

      <Popover open={open} onClose={onClose} className="bottom-full left-0 mb-2 w-[272px]">
        <MenuItem icon={<Layers3 className="h-4 w-4" />} onClick={() => pick(GEOTIFF)}>
          GeoTIFF / TIFF
        </MenuItem>
        <MenuItem icon={<Image className="h-4 w-4" />} onClick={() => pick(PLAIN)}>
          Image (PNG, JPEG)
        </MenuItem>
        <span className="my-1 block h-px bg-line" />
        <MenuItem
          icon={<FileText className="h-4 w-4" />}
          disabled
          note="rasters only"
        >
          PDF
        </MenuItem>
        <MenuItem icon={<Database className="h-4 w-4" />} disabled note="rasters only">
          Other data
        </MenuItem>
        <p className="border-t border-line px-3 pt-2 pb-1 text-[11px] leading-relaxed text-faint">
          The agent analyses raster imagery. A GeoTIFF with a CRS is placed on the map; anything else opens in the image
          viewer.
        </p>
      </Popover>

      {busy && (
        <div className="absolute -top-9 left-0 flex items-center gap-1.5 rounded-full border border-line bg-surface px-2.5 py-1 text-[11px] text-muted shadow-[var(--shadow-sm)]">
          <Spinner className="h-3 w-3" />
          Reading imagery
        </div>
      )}
    </>
  );
}

export { FileImage };
