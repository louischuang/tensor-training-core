from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

from tensor_training_core.utils.paths import resolve_repo_path


def draw_detection_overlays(
    image_path: str | Path,
    output_path: str | Path,
    detections: list[dict[str, object]],
) -> Path:
    image = Image.open(resolve_repo_path(image_path)).convert("RGB")
    width, height = image.size
    draw = ImageDraw.Draw(image)
    palette = [
        (255, 80, 0),
        (0, 150, 255),
        (40, 200, 120),
        (255, 190, 0),
        (190, 60, 255),
    ]

    for index, detection in enumerate(detections):
        x, y, w, h = detection["bbox_xywh_norm"]
        x0 = max(0, min(width, int(x * width)))
        y0 = max(0, min(height, int(y * height)))
        x1 = max(0, min(width, int((x + w) * width)))
        y1 = max(0, min(height, int((y + h) * height)))
        color = palette[index % len(palette)]
        draw.rectangle((x0, y0, x1, y1), outline=color, width=4)

        label_text = f"{detection['label']} {detection['score']:.3f}"
        text_y = y0 - 18 if y0 > 20 else y0 + 4
        draw.rectangle((x0, text_y, min(width, x0 + 200), min(height, text_y + 18)), fill=color)
        draw.text((x0 + 4, text_y + 2), label_text, fill=(255, 255, 255))

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_file)
    return output_file
