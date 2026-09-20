/** Geometry of a drawn area: what a finished shape encloses, and where a drag is anchored. */

import { describe, expect, it } from "vitest";
import { anchoredEvent, drawnArea, featureBounds, MODE_NAMES } from "./aoiGeometry";

const polygon = (ring: [number, number][]): GeoJSON.Feature => ({
  type: "Feature",
  properties: {},
  geometry: { type: "Polygon", coordinates: [ring] },
});

/** A rectangle as Terra Draw finishes one: four corners, closed back to the first. */
const rectangle = (west: number, south: number, east: number, north: number) =>
  polygon([
    [west, south],
    [east, south],
    [east, north],
    [west, north],
    [west, south],
  ]);

/** A circle as Terra Draw approximates one: many vertices around a centre. */
const circle = (lon: number, lat: number, radius: number, segments = 64) =>
  polygon(
    Array.from({ length: segments + 1 }, (_, k) => {
      const angle = (2 * Math.PI * (k % segments)) / segments;
      return [lon + radius * Math.cos(angle), lat + radius * Math.sin(angle)] as [number, number];
    }),
  );

describe("featureBounds", () => {
  it("bounds a rectangle", () => {
    expect(featureBounds(rectangle(10, 50, 11, 51))).toEqual([10, 50, 11, 51]);
  });

  it("bounds a circle by its extent", () => {
    const [west, south, east, north] = featureBounds(circle(10, 50, 0.5));
    expect(west).toBeCloseTo(9.5, 6);
    expect(east).toBeCloseTo(10.5, 6);
    expect(south).toBeCloseTo(49.5, 6);
    expect(north).toBeCloseTo(50.5, 6);
  });

  it("returns an empty range for a geometry with no coordinates", () => {
    const empty = { type: "Feature", properties: {}, geometry: { type: "Polygon", coordinates: [] } } as GeoJSON.Feature;
    expect(featureBounds(empty).every(Number.isFinite)).toBe(false);
  });
});

describe("drawnArea", () => {
  it("accepts a rectangle drag and keeps the feature with its bounds", () => {
    const feature = rectangle(10, 50, 11, 51);
    expect(drawnArea(feature)).toEqual({ feature, bounds: [10, 50, 11, 51] });
  });

  it("accepts a circle drag", () => {
    expect(drawnArea(circle(10, 50, 0.25))).not.toBeNull();
  });

  it("discards a drag that ended where it began", () => {
    expect(drawnArea(rectangle(10, 50, 10, 50))).toBeNull();
  });

  it("discards a shape with no width or no height", () => {
    expect(drawnArea(rectangle(10, 50, 10, 51))).toBeNull(); // a vertical line
    expect(drawnArea(rectangle(10, 50, 11, 50))).toBeNull(); // a horizontal line
  });

  it("discards a geometry it cannot bound", () => {
    const point = { type: "Feature", properties: {}, geometry: { type: "GeometryCollection" } } as unknown as GeoJSON.Feature;
    expect(drawnArea(point)).toBeNull();
  });
});

describe("anchoredEvent", () => {
  const event = { lng: 2, lat: 3, containerX: 20, containerY: 30, button: "left", heldKeys: [] } as never;

  it("starts the shape where the pointer went down, not where the drag was detected", () => {
    const anchored = anchoredEvent(event, { lng: 1, lat: 1.5, containerX: 10, containerY: 15 });
    expect(anchored).toMatchObject({ lng: 1, lat: 1.5, containerX: 10, containerY: 15 });
  });

  it("keeps the rest of the event", () => {
    expect(anchoredEvent(event, { lng: 1, lat: 1, containerX: 1, containerY: 1 })).toMatchObject({ button: "left" });
  });

  it("passes the event through when no press was recorded", () => {
    expect(anchoredEvent(event, null)).toBe(event);
  });
});

describe("the area tools", () => {
  it("are the three that enclose an area, with no point tool (D-025)", () => {
    expect(Object.keys(MODE_NAMES).sort()).toEqual(["circle", "polygon", "rectangle"]);
  });
});
