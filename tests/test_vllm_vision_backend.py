
from pathlib import Path

from app.image_processing import ImageProcessor
from app.identification import BirdIdentifier
from app.output import ResultWriter
from inference.vllm_backend import VLLMBackend


def main():
    # Define project paths
    project_root = Path(__file__).resolve().parents[1]
    original_image = project_root / "data" / "input" / "IMG_0374.jpg"
    processed_dir = project_root / "data" / "processed"
    results_dir = project_root / "data" / "results"

    print("\n" + "=" * 60)
    print("BIRD IDENTIFICATION PIPELINE")
    print("=" * 60)

    # Step 1: Process image
    print("\n[1/4] Processing image...")

    processor = ImageProcessor(
        max_dimension=1280,
        jpeg_quality=90,
    )

    processed_image = processor.process(
        image_path=original_image,
        output_dir=processed_dir,
    )

    print(f"Original image:  {processed_image.original_path}")
    print(f"Original size:   {processed_image.original_size}")
    print(f"Processed image: {processed_image.processed_path}")
    print(f"Processed size:  {processed_image.processed_size}")

    # Step 2: Set observation information
    location = None
    observation_date = None

    # Step 3: Run bird identification
    print("\n[2/4] Identifying bird...")

    backend = VLLMBackend(
        model_name="bird-vl",
        base_url="http://localhost:8000/v1",
    )

    identifier = BirdIdentifier(backend=backend)

    result = identifier.identify(
        image_path=processed_image.processed_path,
        location=location,
        observation_date=observation_date,
        max_tokens=500,
        temperature=0.2,
    )

    # Step 4: Display and save result
    print("\n[3/4] Displaying result...")

    writer = ResultWriter()
    writer.display(result)

    print("[4/4] Saving result...")

    saved_path = writer.save_json(
        response=result,
        original_image=processed_image.original_path,
        processed_image=processed_image.processed_path,
        model_name="bird-vl",
        output_dir=results_dir,
        location=location,
        observation_date=observation_date,
    )

    print(f"\nResult saved to: {saved_path}")

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
