from __future__ import annotations

from pydantic import BaseModel


class ManifestRecord(BaseModel):
    image_id: int
    image_path: str
    width: int
    height: int
    category_id: int
    category_name: str
    bbox_xywh: list[float]
