from __future__ import annotations

from PIL import Image

from utils.math_utils import clip01
from utils.types import StreamScore


class MetadataProvenanceAnalyzer:
    """Lightweight EXIF/software provenance analyzer.

    Missing metadata is treated as weak evidence only; many genuine web images
    are stripped during upload.
    """

    name = "forensic_metadata_provenance"

    camera_tags = {271, 272, 305, 306, 33434, 33437, 34855, 36867, 37386}
    suspicious_software_tokens = {
        "stable diffusion",
        "midjourney",
        "dall-e",
        "dalle",
        "firefly",
        "comfyui",
        "automatic1111",
        "invokeai",
        "fooocus",
        "flux",
        "generative",
    }

    def analyze(self, image: Image.Image) -> StreamScore:
        exif = {}
        try:
            exif = dict(image.getexif() or {})
        except Exception:
            exif = {}

        info = {str(key).lower(): str(value).lower() for key, value in image.info.items()}
        software = str(exif.get(305, "")).lower() or info.get("software", "")
        camera_tag_count = sum(1 for tag in self.camera_tags if tag in exif)
        has_camera_trace = camera_tag_count >= 3
        has_any_exif = bool(exif)
        suspicious_software = any(token in software for token in self.suspicious_software_tokens)

        missing_metadata_score = 0.28 if not has_any_exif else 0.12 if not has_camera_trace else 0.0
        software_score = 0.85 if suspicious_software else 0.0
        editing_score = 0.20 if software and not suspicious_software and not has_camera_trace else 0.0
        score = clip01(max(software_score, missing_metadata_score + editing_score))

        return StreamScore(
            self.name,
            score,
            {
                "has_exif": has_any_exif,
                "camera_tag_count": camera_tag_count,
                "has_camera_trace": has_camera_trace,
                "software": software,
                "suspicious_software": suspicious_software,
            },
        )
