"""Fetch a small bi-temporal Sentinel-2 L2A pair for a demo scenario [R1-REQ].

Uses the public Earth Search STAC API (Element 84, AWS open data; no login). Only the requested window is read
from cloud-optimised GeoTIFFs. Both dates are resampled onto ONE common UTM grid, so the pair passes validation.
Output: <out>/before.tif and <out>/after.tif (bands B02, B03, B04, B08 with descriptions), plus source.json.
Attribution: "Contains modified Copernicus Sentinel data [year]".

Example (Navi Mumbai airport construction site):
  python scripts/fetch_sentinel2_pair.py --bbox 73.045 18.975 73.095 19.020 \
      --before 2018-01-01/2018-03-31 --after 2025-01-01/2025-03-31 --out demo/examples/navi_mumbai_change
"""

import argparse
import json
import math
import urllib.request
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform_bounds

STAC_SEARCH = "https://earth-search.aws.element84.com/v1/search"
BANDS = {"B02": "blue", "B03": "green", "B04": "red", "B08": "nir"}  # band name -> Earth Search v1 asset key


def rfc3339(date_range):
    start, end = date_range.split("/")
    return f"{start}T00:00:00Z/{end}T23:59:59Z" if "T" not in date_range else date_range


def search(bbox, date_range, max_cloud):
    body = {"collections": ["sentinel-2-l2a"], "bbox": bbox, "datetime": rfc3339(date_range), "limit": 50,
            "query": {"eo:cloud_cover": {"lt": max_cloud}}}
    request = urllib.request.Request(STAC_SEARCH, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=60) as response:
        items = json.load(response)["features"]
    if not items:
        raise SystemExit(f"no scene with cloud cover < {max_cloud}% in {date_range}")
    return sorted(items, key=lambda item: item["properties"]["eo:cloud_cover"])


def utm_crs(lon, lat):
    return f"EPSG:{(32600 if lat >= 0 else 32700) + int((lon + 180) // 6) + 1}"


def read_window(item, crs, transform, width, height):
    bands = []
    for key in BANDS:
        href = item["assets"][BANDS[key]]["href"]
        with rasterio.open(href) as src, WarpedVRT(src, crs=crs, transform=transform, width=width, height=height,
                                                    resampling=Resampling.bilinear) as vrt:
            bands.append(vrt.read(1))
    return np.stack(bands)


def write(path, data, crs, transform, item):
    with rasterio.open(path, "w", driver="GTiff", height=data.shape[1], width=data.shape[2], count=data.shape[0],
                       dtype=data.dtype, crs=crs, transform=transform, compress="deflate") as dst:
        dst.write(data)
        for i, name in enumerate(BANDS):
            dst.set_band_description(i + 1, name)
        dst.update_tags(ACQUISITION=item["properties"]["datetime"], SOURCE=item["id"],
                        ATTRIBUTION="Contains modified Copernicus Sentinel data")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bbox", nargs=4, type=float, required=True, metavar=("LON_MIN", "LAT_MIN", "LON_MAX", "LAT_MAX"))
    parser.add_argument("--before", required=True, help="date range, e.g. 2018-01-01/2018-03-31")
    parser.add_argument("--after", required=True, help="date range, e.g. 2025-01-01/2025-03-31")
    parser.add_argument("--max-cloud", type=float, default=5)
    parser.add_argument("--resolution", type=float, default=10.0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    lon_min, lat_min, lon_max, lat_max = args.bbox
    crs = utm_crs((lon_min + lon_max) / 2, (lat_min + lat_max) / 2)
    left, bottom, right, top = transform_bounds("EPSG:4326", crs, lon_min, lat_min, lon_max, lat_max)
    width, height = math.ceil((right - left) / args.resolution), math.ceil((top - bottom) / args.resolution)
    transform = from_origin(left, top, args.resolution, args.resolution)
    args.out.mkdir(parents=True, exist_ok=True)

    sources = {}
    for name, date_range in (("before", args.before), ("after", args.after)):
        for item in search(args.bbox, date_range, args.max_cloud)[:6]:  # least cloudy first
            data = read_window(item, crs, transform, width, height)
            empty = float((data == 0).all(axis=0).mean())
            print(f"{name}: {item['id']} ({item['properties']['datetime'][:10]}, cloud {item['properties']['eo:cloud_cover']:.1f}%, "
                  f"empty {empty:.1%})")
            if empty < 0.01:  # the tile must fully cover the window
                break
        else:
            raise SystemExit(f"no fully covering scene found for {name}")
        write(args.out / f"{name}.tif", data, crs, transform, item)
        sources[name] = {"id": item["id"], "datetime": item["properties"]["datetime"],
                         "cloud_cover": item["properties"]["eo:cloud_cover"]}
    sources.update(bbox=args.bbox, crs=crs, size=[width, height], attribution="Contains modified Copernicus Sentinel data")
    (args.out / "source.json").write_text(json.dumps(sources, indent=2), encoding="utf-8")
    print(f"wrote {width}x{height} px pair to {args.out}")


if __name__ == "__main__":
    main()
