# Demo UI Notes

> **Development-only.** Kept for future development reference. It is not linked from the app and must not
> appear in user-facing UI. It may be removed during final repository cleanup.

## Main Map Interface (2026-09-27)

The primary map interface should remain clean and focused on the user's interaction with SatQuery.

The informational explanation above the AI command bar was intentionally removed from the main interface. It
exposed implementation details such as Sentinel-2, Sentinel-1 SAR, Copernicus Data Space, GeoTIFF handling and
weather routing before the user asked a question.

For the SIH demo, the preferred interaction is:

**Select an area → see example capabilities → ask a natural-language question.**

Technical implementation and data-source information remains available through Help, the Details drawer, result
provenance and result metadata, rather than occupying the main idle interface. No capability was removed.

With an area selected, the command bar placeholder reads:

"Ask about this area..."

(three ASCII dots, matching every other placeholder in the web client).

### What changed, and what did not

- Removed: the "Area selected…" box above the command bar (`web/src/command/AICommandBar.tsx`).
- Changed: the command bar placeholder, whenever an area is selected, to the text above.
- Kept: the example-question chips above the command bar; error messages; the separate note shown only when a
  circle or polygon is drawn (imagery retrieval needs a rectangle), which was out of scope for this change.
- Unchanged: all backend behaviour, including imagery retrieval, Sentinel-1/SAR, temporal analysis, weather,
  GeoTIFF upload, area selection and the agent.
