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
    """Return a page image normalized for reading, using Azure DI page angle.

    Azure Document Intelligence exposes a page-level ``angle`` when the scanned
    text is rotated. In these money-order batches, several real *front* pages are
    scanned upside down. GPT vision can often read them, but it regularly drops
    amount words/cents on inverted money orders, e.g. 442.00 -> 400.00 or
    525.50 -> 525.00. Normalize only when Azure reports a clear right-angle
    rotation so already-upright pages are not damaged.
    """
    if angle is None:
        return image
    try:
        rounded = int(round(float(angle))) % 360
    except (TypeError, ValueError):
        return image

    # DI angles are the observed page orientation. Rotate in the opposite direction
    # for 90/270, and 180 for upside-down pages.
    if rounded in range(80, 101):
        return image.rotate(-90, expand=True)
    if rounded in range(170, 191):
        return image.rotate(180, expand=True)
    if rounded in range(260, 281):
        return image.rotate(90, expand=True)
    return image
