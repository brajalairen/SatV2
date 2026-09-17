"""BigEarthNet v2 Sentinel-1 land-cover classifier. [R1-OPT], domain-gated (D-015). Not yet in the tool registry.

Ported from the former remote_sensing/sar/inference.py (author: rolenson26).
Model: BIFOLD-BigEarthNetv2-0/resnet18-s1-v0.2.0 (MIT; 2 channels, 120x120, 19 classes). It needs the
`reben_publication` code from git.tu-berlin.de/rsim/reben-training-scripts plus `configilm`; neither is pip-installable here.
UNVERIFIED: the channel order (VV, VH) and that the label order below matches class indices "0".."18".
Trained on European Sentinel-1 patches: do not trust it on RISAT or on data that is not ~10 m VV/VH.
"""

import numpy as np

from satquery.imaging import RasterImage

MODEL_ID = "BIFOLD-BigEarthNetv2-0/resnet18-s1-v0.2.0"
LABELS = [
    "Agro-forestry areas", "Arable land", "Beaches, dunes, sands", "Broad-leaved forest", "Coastal wetlands",
    "Complex cultivation patterns", "Coniferous forest", "Industrial or commercial units", "Inland waters",
    "Inland wetlands", "Land principally occupied by agriculture, with significant areas of natural vegetation",
    "Marine waters", "Mixed forest", "Moors, heathland and sclerophyllous vegetation",
    "Natural grassland and sparsely vegetated areas", "Pastures", "Permanent crops", "Transitional woodland, shrub",
    "Urban fabric",
]
S1_MEAN = np.array([-12.643863677978516, -19.352558135986328], dtype=np.float32)  # BigEarthNet v2 VV, VH
S1_STD = np.array([5.133493900299072, 5.590505599975586], dtype=np.float32)


def domain_check(image: RasterImage) -> str | None:
    """Return a reason to skip, or None if the image resembles the training domain."""
    names = {n.lower() for n in image.band_names}
    if image.modality != "sar" or not {"vv", "vh"} <= names:
        return "classifier expects Sentinel-1 VV+VH bands (named in the GeoTIFF)"
    if image.transform and not 5 <= abs(image.transform[0]) <= 20:
        return "classifier was trained on ~10 m pixels"
    return None


class S1LandCoverClassifier:
    def __init__(self):
        self.model = None

    def load(self):
        if self.model is None:
            from reben_publication.BigEarthNetv2_0_ImageClassifier import BigEarthNetv2_0_ImageClassifier
            self.model = BigEarthNetv2_0_ImageClassifier.from_pretrained(MODEL_ID).eval()
        return self

    def predict(self, image: RasterImage) -> list[dict]:
        import torch

        self.load()
        stack = np.stack([image.band("vv"), image.band("vh")]).astype(np.float32)
        normalized = np.nan_to_num((stack - S1_MEAN[:, None, None]) / S1_STD[:, None, None])
        with torch.no_grad():
            probabilities = torch.sigmoid(self.model(torch.from_numpy(normalized).unsqueeze(0)))[0]
        return sorted(({"label": label, "probability": float(p)} for label, p in zip(LABELS, probabilities)),
                      key=lambda r: r["probability"], reverse=True)
