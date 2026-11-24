import json
from datetime import datetime
from pathlib import Path
from app.validation import IdentificationResult

class ResultWriter:
    """
    Displays and saves bird identification results.

    Responsible for presenting and storing results.
    Does not communicate with inference backends or interpret
    the model's identification.
    """

    def display(self, identification_result: IdentificationResult) -> None:
        """Print the identification result in a readable terminal format."""
        print("\n" + "=" * 60)
        print("BIRD IDENTIFICATION RESULT")
        print("=" * 60)
        print(json.dumps(identification_result.model_dump(), indent=4, ensure_ascii=False))
        print("=" * 60 + "\n")

    def save_json(
        self,
        *,
        identification_result: IdentificationResult,
        original_image: str | Path,
        processed_image: str | Path,
        model_name: str,
        output_dir: str | Path,
        location: str | None = None,
        observation_date: str | None = None,
    ) -> Path:
        """Save an identification result and its metadata as JSON."""

        original_image = Path(original_image).resolve()
        processed_image = Path(processed_image).resolve()
        output_dir = Path(output_dir)

        output_dir.mkdir(parents=True, exist_ok=True)

        created_at = datetime.now().astimezone()

        result = {
            "created_at": created_at.isoformat(timespec="seconds"),
            "original_image": str(original_image),
            "processed_image": str(processed_image),
            "location": location,
            "observation_date": observation_date,
            "model": model_name,
            "identification": {
                "response": identification_result.model_dump(mode="json")
            },
        }

        timestamp = created_at.strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{original_image.stem}_{timestamp}.json"
        output_path = output_dir / filename

        with output_path.open("w", encoding="utf-8") as file:
            json.dump(result, file, indent=4, ensure_ascii=False)

        return output_path
