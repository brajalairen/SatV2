/** Which logo variant each map gets: dark ink only on the one light map. */

import { describe, expect, it } from "vitest";
import { logoVariant } from "./BrandMark";

describe("the logo variant follows the map under it", () => {
  it("dark ink on the Standard basemap in light mode, the only light map", () => {
    expect(logoVariant("standard", "light")).toBe("dark");
  });

  it("white ink on imagery basemaps, whatever the theme", () => {
    for (const theme of ["light", "dark"] as const) {
      expect(logoVariant("satellite", theme)).toBe("white");
      expect(logoVariant("hybrid", theme)).toBe("white");
    }
  });

  it("white ink on the Standard basemap in dark mode", () => {
    expect(logoVariant("standard", "dark")).toBe("white");
  });
});
