
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from app.output import ResultWriter


class TestResultWriter(unittest.TestCase):

    def setUp(self):
        self.writer = ResultWriter()
        self.response = (
            "The most likely species is Barn Swallow "
            "(Hirundo rustica)."
        )

    def test_display(self):
        """Check that the response is printed to the terminal."""
        output = StringIO()

        with redirect_stdout(output):
            self.writer.display(self.response)

        self.assertIn(self.response, output.getvalue())
        self.assertIn("BIRD IDENTIFICATION RESULT", output.getvalue())

    def test_save_json(self):
        """Check that a result is saved with the expected metadata."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "results"

            output_path = self.writer.save_json(
                response=self.response,
                original_image="IMG_0374.jpg",
                processed_image="IMG_0374_processed.jpg",
                model_name="bird-vl",
                output_dir=output_dir,
                location="Alexandria, Virginia",
                observation_date="2026-11-08",
            )

            self.assertTrue(output_path.is_file())
            self.assertEqual(output_path.suffix, ".json")

            with output_path.open("r", encoding="utf-8") as file:
                result = json.load(file)

            self.assertEqual(
                result["identification"]["response"],
                self.response,
            )
            self.assertEqual(result["model"], "bird-vl")
            self.assertEqual(
                result["location"],
                "Alexandria, Virginia",
            )
            self.assertEqual(
                result["observation_date"],
                "2026-11-08",
            )
            self.assertIn("created_at", result)
            self.assertIn("original_image", result)
            self.assertIn("processed_image", result)

    def test_unique_filenames(self):
        """Check that consecutive results don't overwrite each other."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)

            arguments = {
                "response": self.response,
                "original_image": "IMG_0374.jpg",
                "processed_image": "IMG_0374_processed.jpg",
                "model_name": "bird-vl",
                "output_dir": output_dir,
            }

            first_path = self.writer.save_json(**arguments)
            second_path = self.writer.save_json(**arguments)

            self.assertNotEqual(first_path, second_path)
            self.assertTrue(first_path.is_file())
            self.assertTrue(second_path.is_file())


if __name__ == "__main__":
    unittest.main()
