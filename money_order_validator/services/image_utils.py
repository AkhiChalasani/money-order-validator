from __future__ import annotations

from PIL import Image, ImageChops, ImageOps


def crop_to_content(image: Image.Image, margin: int = 30) -> Image.Image:
    """Crop large white borders while preserving page/instrument content.

    This reduces image-token usage on scans where the instrument occupies only the top-left
    quadrant. If no reliable bounding box is found, returns the original image.
    """
    img = image.convert("RGB")
    # Difference against a white canvas works well for scanned PDFs.
    bg = Image.new("RGB", img.size, (255, 255, 255))
    diff = ImageChops.difference(img, bg)
    gray = ImageOps.grayscale(diff)
    # Ignore very light paper noise.
    mask = gray.point(lambda p: 255 if p > 18 else 0)
    bbox = mask.getbbox()
    if not bbox:
        return img
    left, top, right, bottom = bbox
    left = max(0, left - margin)
    top = max(0, top - margin)
    right = min(img.width, right + margin)
    bottom = min(img.height, bottom + margin)
    # Avoid over-cropping tiny/noisy boxes.
    if (right - left) < img.width * 0.15 or (bottom - top) < img.height * 0.10:
        return img
    return img.crop((left, top, right, bottom))


def maybe_rotate_for_reading(image: Image.Image, angle: float | None) -> Image.Image:
    """DI angle is informational; GPT vision handles rotation well.

    Only rotate for obvious 90-degree page angle values. Keeping this conservative prevents
    damaging already-readable scans.
    """
    if angle is None:
        return image
    rounded = int(round(angle)) % 360
    if rounded in (90, 270):
        return image.rotate(-rounded, expand=True)
    return image
