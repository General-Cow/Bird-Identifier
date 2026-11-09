
import argparse
from pathlib import Path

from app.image_processing import ImageProcessor
from app.identification import BirdIdentifier
from app.output import ResultWriter
from inference.vllm_backend import VLLMBackend


def parse_arguments():
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Identify bird species in an image using a local VLM."
    )

    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to the bird image to identify.",
    )

    parser.add_argument(
        "--location",
        type=str,
        default=None,
        help="Location where the bird was observed.",
    )

    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Date the bird was observed (YYYY-MM-DD).",
    )

    parser.add_argument(
        "--max-dimension",
        type=int,
        default=1280,
        help="Maximum width or height of the processed image. Default: 1280.",
    )

    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=90,
        help="JPEG quality for the processed image (1-100). Default: 90.",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="bird-vl",
        help="Name of the model served by vLLM. Default: bird-vl.",
    )

    parser.add_argument(
        "--base-url",
        type=str,
        default="http://localhost:8000/v1",
        help="vLLM API endpoint.",
    )

    parser.add_argument(
        "--max-tokens",
        type=int,
        default=500,
        help="Maximum number of tokens in the model response.",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Model sampling temperature.",
    )

    return parser.parse_args()


def main():
    args = parse_arguments()

    # Establish project paths
    project_root = Path(__file__).resolve().parents[1]

    original_image = Path(args.image).expanduser().resolve()

    processed_dir = project_root / "data" / "processed"
    results_dir = project_root / "data" / "results"

    print("\n" + "=" * 60)
    print("BIRD IDENTIFICATION")
    print("=" * 60)

    # Step 1: Process image
    print("\n[1/4] Processing image...")

    processor = ImageProcessor(
        max_dimension=args.max_dimension,
        jpeg_quality=args.jpeg_quality,
    )

    processed_image = processor.process(
        image_path=original_image,
        output_dir=processed_dir,
    )

    print(f"Original image:  {processed_image.original_path}")
    print(f"Original size:   {processed_image.original_size}")
    print(f"Processed image: {processed_image.processed_path}")
    print(f"Processed size:  {processed_image.processed_size}")

    # Step 2: Configure inference backend
    print("\n[2/4] Connecting to inference backend...")

    backend = VLLMBackend(
        model_name=args.model,
        base_url=args.base_url,
    )

    identifier = BirdIdentifier(backend=backend)

    # Step 3: Identify bird
    print("\n[3/4] Identifying bird...")

    result = identifier.identify(
        image_path=processed_image.processed_path,
        location=args.location,
        observation_date=args.date,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )

    # Step 4: Display and save result
    print("\n[4/4] Saving results...")

    writer = ResultWriter()

    writer.display(result)

    saved_path = writer.save_json(
        response=result,
        original_image=processed_image.original_path,
        processed_image=processed_image.processed_path,
        model_name=args.model,
        output_dir=results_dir,
        location=args.location,
        observation_date=args.date,
    )

    print(f"Result saved to: {saved_path}")

    print("\n" + "=" * 60)
    print("IDENTIFICATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
