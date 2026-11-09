from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageOps


@dataclass
class ProcessedImage:
    """Information about an original image and its processed copy."""

    original_path: Path
    processed_path: Path
    original_format: str
    original_size: tuple[int, int]
    processed_size: tuple[int, int]
    date_taken: str | None


class ImageProcessor:
    """
    Loads and prepares images for model inference.

    The original image is never modified. A resized JPEG copy is
    saved in the output directory.
    """

    SUPPORTED_FORMATS = {"JPEG", "PNG", "WEBP"}

    def __init__(
        self,
        max_dimension: int = 2048,
        jpeg_quality: int = 90,
    ):
        if max_dimension <= 0:
            raise ValueError("max_dimension must be greater than zero.")

        if not 1 <= jpeg_quality <= 100:
            raise ValueError("jpeg_quality must be between 1 and 100.")

        self.max_dimension = max_dimension
        self.jpeg_quality = jpeg_quality

    def process(
        self,
        image_path: str | Path,
        output_dir: str | Path,
    ) -> ProcessedImage:
        """
        Validate, orient, resize, and save an image.

        Args:
            image_path: Path to the original image.
            output_dir: Directory where the processed copy is saved.

        Returns:
            A ProcessedImage containing paths and image metadata.
        """

        image_path = Path(image_path)
        output_dir = Path(output_dir)

        if not image_path.is_file():
            raise FileNotFoundError(f"Image not found: {image_path}")

        output_dir.mkdir(parents=True, exist_ok=True)

        with Image.open(image_path) as image:
            original_format = image.format

            if original_format not in self.SUPPORTED_FORMATS:
                raise ValueError(
                    f"Unsupported image format: {original_format}. "
                    f"Supported formats: {', '.join(sorted(self.SUPPORTED_FORMATS))}"
                )

            original_size = image.size

            # Capture the date before converting or processing the image.
            exif = image.getexif()
            date_taken = exif.get(36867) or exif.get(306)

            # Correct rotation based on EXIF orientation.
            image = ImageOps.exif_transpose(image)

            # Resize only if either dimension exceeds the limit.
            image.thumbnail(
                (self.max_dimension, self.max_dimension),
                Image.Resampling.LANCZOS,
            )

            # JPEG does not support transparency, so composite transparent
            # images against a white background before saving.
            if image.mode in ("RGBA", "LA") or (
                image.mode == "P" and "transparency" in image.info
            ):
                image = image.convert("RGBA")
                background = Image.new("RGB", image.size, "white")
                background.paste(image, mask=image.getchannel("A"))
                image = background
            else:
                image = image.convert("RGB")

            processed_size = image.size

            processed_path = output_dir / f"{image_path.stem}_processed.jpg"

            image.save(
                processed_path,
                format="JPEG",
                quality=self.jpeg_quality,
                optimize=True,
            )

        return ProcessedImage(
            original_path=image_path,
            processed_path=processed_path,
            original_format=original_format,
            original_size=original_size,
            processed_size=processed_size,
            date_taken=date_taken,
        )