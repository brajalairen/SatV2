/** Basemap styles. Pure configuration: no map is created, so no MapLibre instance is needed. */

import { describe, expect, it } from "vitest";
import type { StyleSpecification } from "maplibre-gl";
import { BASEMAPS, basemapStyle, basemapUrl, isBasemapId, styleKey } from "./basemap";

const styleObject = (basemap: "satellite" | "hybrid") => basemapStyle(basemap, "light") as StyleSpecification;

describe("the basemaps offered", () => {
  it("are Standard, Satellite and Hybrid; there is no Terrain source, so no Terrain option", () => {
    expect(BASEMAPS.map((b) => b.label)).toEqual(["Standard", "Satellite", "Hybrid"]);
    expect(isBasemapId("terrain")).toBe(false);
  });

  it("keeps Standard exactly as before: OpenFreeMap, following the theme", () => {
    expect(basemapStyle("standard", "light")).toBe(basemapUrl("light"));
    expect(basemapStyle("standard", "dark")).toBe(basemapUrl("dark"));
    expect(basemapUrl("light")).toBe("https://tiles.openfreemap.org/styles/positron");
    expect(basemapUrl("dark")).toBe("https://tiles.openfreemap.org/styles/dark");
  });

  it("draws Satellite from Esri World Imagery alone", () => {
    const style = styleObject("satellite");
    expect(Object.keys(style.sources)).toEqual(["imagery"]);
    expect(JSON.stringify(style.sources)).toContain("/World_Imagery/MapServer/tile/{z}/{y}/{x}");
  });

  it("addresses Esri tiles row before column, as the service expects", () => {
    // {z}/{x}/{y} would silently fetch the wrong place: Dubai is z10 row 438, column 668.
    for (const basemap of ["satellite", "hybrid"] as const) {
      expect(JSON.stringify(styleObject(basemap).sources)).not.toContain("{z}/{x}/{y}");
    }
  });

  it("lays roads, then place names, over the imagery for Hybrid", () => {
    const order = styleObject("hybrid").layers.map((layer) => layer.id);
    expect(order.indexOf("basemap-imagery")).toBeGreaterThan(-1);
    expect(order.indexOf("basemap-imagery")).toBeLessThan(order.indexOf("basemap-roads"));
    expect(order.indexOf("basemap-roads")).toBeLessThan(order.indexOf("basemap-places"));
  });

  it("needs no API key, and credits every imagery source", () => {
    for (const basemap of ["satellite", "hybrid"] as const) {
      const style = styleObject(basemap);
      expect(JSON.stringify(style)).not.toMatch(/token|apikey|api_key|[?&]key=/i);
      for (const source of Object.values(style.sources)) {
        expect("attribution" in source && source.attribution).toMatch(/Esri/);
      }
    }
  });

  it("never uses an id the overlay, area or drawing reconciler would claim as its own", () => {
    // MapView removes stale sources prefixed raster-/overlay-, owns "aoi", and Terra Draw uses "td-".
    for (const basemap of ["satellite", "hybrid"] as const) {
      const style = styleObject(basemap);
      const ids = [...Object.keys(style.sources), ...style.layers.map((layer) => layer.id)];
      for (const id of ids) expect(id).not.toMatch(/^(raster-|overlay-|aoi|td-)/);
    }
  });
});

describe("when the map reloads its style", () => {
  it("reloads for Standard when the theme changes", () => {
    expect(styleKey("standard", "light")).not.toBe(styleKey("standard", "dark"));
  });

  it("does not reload imagery when only the theme changes", () => {
    expect(styleKey("satellite", "light")).toBe(styleKey("satellite", "dark"));
    expect(styleKey("hybrid", "light")).toBe(styleKey("hybrid", "dark"));
  });

  it("reloads whenever the basemap changes", () => {
    const keys = BASEMAPS.map((b) => styleKey(b.id, "light"));
    expect(new Set(keys).size).toBe(BASEMAPS.length);
  });
});
