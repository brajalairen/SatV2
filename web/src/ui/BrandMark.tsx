/** The SatQuery AI wordmark, top-left, directly above the sidebar's control rail.
 *
 *  One transparent asset (src/assets/satquery-logo.png, 2172 x 724, used exactly as supplied) holds
 *  two variants: white ink on top, for dark maps, and dark ink below, for light ones. CSS shows only
 *  the variant that suits the map under it: there is no second file, and nothing behind the logo (no
 *  background, border or shadow). Only the Standard basemap in light mode is a light map; Satellite,
 *  Hybrid and everything in dark mode get the white variant. It switches the moment the theme or
 *  basemap changes, with no new download: both variants are the same file.
 *
 *  It floats above the map, not in it, so panning, zooming and basemap changes never move it. It
 *  takes no pointer events, so the map and area drawing work underneath it. It shares the rail's
 *  layer (z-30) and is rendered before the sidebar, so a panel that opens over it covers it. */

import logo from "../assets/satquery-logo.png";
import type { BasemapId } from "../map/basemap";
import { useAppStore } from "../state/useAppStore";

const IMAGE = { width: 2172, height: 724 };
// One crop size for both variants, so switching never moves or resizes the logo. Each variant's
// wordmark (white: 438..1818 x 74..341, dark: 438..1802 x 415..673 in the asset) sits centred in it
// with 8 px of the asset's own transparency around it.
const CROP = { x: 430, width: 1396, height: 283 };
const VARIANT_TOP = { white: 66, dark: 402 };

export type LogoVariant = keyof typeof VARIANT_TOP;

/** Dark ink where the map is light (Standard in light mode); white ink everywhere else. */
export function logoVariant(basemap: BasemapId, theme: "light" | "dark"): LogoVariant {
  return basemap === "standard" && theme === "light" ? "dark" : "white";
}

export function BrandMark() {
  const variant = useAppStore((s) => logoVariant(s.basemap, s.theme));
  return (
    <div
      // 28 px tall (about 138 px wide), 24 px on small screens. top-6 centres it on the top control
      // row and leaves a small gap above the toolbar, which stays where it is (Sidebar.tsx).
      className="pointer-events-none absolute top-6 left-4 z-30 h-7 overflow-hidden select-none max-sm:h-6"
      style={{ aspectRatio: `${CROP.width} / ${CROP.height}` }}
      data-variant={variant}
    >
      <img
        src={logo}
        alt="SatQuery AI"
        draggable={false}
        className="absolute max-w-none"
        style={{
          width: `${(IMAGE.width / CROP.width) * 100}%`,
          left: `${(-CROP.x / CROP.width) * 100}%`,
          top: `${(-VARIANT_TOP[variant] / CROP.height) * 100}%`,
        }}
      />
    </div>
  );
}
