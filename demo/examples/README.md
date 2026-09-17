# Demo scenarios [R1-REQ]

Small example inputs shown as one-click scenarios in the web app. They are shipped to the Space.

**Rules:**
- The **whole folder stays under ~10 MB**.
- Only commit data whose licence allows redistribution, and credit the source below.
- **Never commit a full dataset here** (D-019).

## Format: `examples.json`
```json
[
  {
    "label": "Cross-modal: water and built-up",
    "images": [
      {"path": "bigearthnet_patch1/s2_bgrn.tif", "modality": "optical"},
      {"path": "bigearthnet_patch1/s1_vv_vh.tif", "modality": "sar"}
    ],
    "query": "Use the optical and SAR images together to identify built-up and water-covered regions."
  }
]
```
`path` is relative to this folder. `acquired` (a `YYYY-MM-DD` date) is optional, but set it for before/after pairs.

## Shipped scenarios (order matches `examples.json` and the UI)
All eight were validated end to end with real Falcon on the RTX 4050 on 2026-09-17: every one returned
status `ok` with no failed steps, except #8, which is *meant* to be rejected.

| # | Capability | Input | Source |
|---|---|---|---|
| 1 | Single-image VQA | 1 optical | `bigearthnet_coast_finland` |
| 2 | Captioning | 1 optical | `bigearthnet_river_austria` |
| 3 | Grounding | 1 optical | `bigearthnet_coast_finland` |
| 4 | Change analysis | 2 optical, dated | `navi_mumbai_change` |
| 5 | Change VQA (comparative) | 2 optical, dated | `navi_mumbai_change` |
| 6 | Optical + SAR | optical + SAR, same grid | `bigearthnet_coast_finland` |
| 7 | Single SAR image | 1 SAR | `bigearthnet_coast_finland` |
| 8 | Input rejection | 2 images on different grids | mixes the two sources above |

The BigEarthNet folders pair S2 with its S1 counterpart using the `s1_name` column of `BigEarthNet.txt.parquet`.
Both pairs were checked to share CRS, transform and shape, so no co-registration is performed (D-008).
Each folder's `text.json` holds that patch's BigEarthNet.txt question/answer rows, which give ground truth to quote.

## Attribution
- BigEarthNet v2.0 / BigEarthNet.txt: CDLA-Permissive-1.0; contains modified Copernicus Sentinel data.
- Copernicus Sentinel exports: "Contains modified Copernicus Sentinel data [year]".
