/** What the upload request carries. Only `fetch` is stubbed: the FormData is the real one sent. */

import { afterEach, describe, expect, it, vi } from "vitest";
import { api } from "./api";

function capture() {
  const sent: FormData[] = [];
  vi.stubGlobal(
    "fetch",
    vi.fn(async (_path: string, init?: RequestInit) => {
      sent.push(init?.body as FormData);
      return new Response(JSON.stringify({ id: "u1" }), { status: 200 });
    }),
  );
  return sent;
}

afterEach(() => vi.unstubAllGlobals());

describe("uploading a raster", () => {
  it("sends no modality by default, so the server reads it from the file's band descriptions", async () => {
    const sent = capture();
    await api.upload(new File(["x"], "s1.tif"));
    expect(sent[0]!.has("modality")).toBe(false);
    expect((sent[0]!.get("file") as File).name).toBe("s1.tif");
  });

  it("still sends a modality the caller chose", async () => {
    const sent = capture();
    await api.upload(new File(["x"], "s1.tif"), "sar", "2024-05-01");
    expect(sent[0]!.get("modality")).toBe("sar");
    expect(sent[0]!.get("acquired")).toBe("2024-05-01");
  });
});
