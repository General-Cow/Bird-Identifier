from app.validation import IdentificationResult
import unittest
from pydantic import ValidationError

class TestIdentificationResult(unittest.TestCase):
    def test_valid_identification(self):
        data = {
            "species": {
                "common_name": "Barn Swallow",
                "scientific_name": "Hirundo rustica"
            },
            "confidence": "high",
            "visual_evidence": [
                "Dark blue-black upper body",
                "Rust-colored underparts",
                "Slender body shape"
            ],
            "alternative_species": [
                {
                    "common_name": "Cliff Swallow",
                    "scientific_name": "Petrochelidon pyrrhonota",
                    "reason": "Similar body shape and coloration"
                }
            ],
            "limitations": [
                "Tail markings are not clearly visible"
            ]
        }

        result = IdentificationResult.model_validate(data)

        self.assertIsInstance(result, IdentificationResult)

    def test_invalid_confidence(self):
        data = {
            "species": {
                "common_name": "Barn Swallow",
                "scientific_name": "Hirundo rustica"
            },
            "confidence": "extremely high",  # Invalid confidence value
            "visual_evidence": [
                "Dark blue-black upper body",
                "Rust-colored underparts",
                "Slender body shape"
            ],
            "alternative_species": [
                {
                    "common_name": "Cliff Swallow",
                    "scientific_name": "Petrochelidon pyrrhonota",
                    "reason": "Similar body shape and coloration"
                }
            ],
            "limitations": [
                "Tail markings are not clearly visible"
            ]
        }

        with self.assertRaises(ValidationError) as context:
            IdentificationResult.model_validate(data)

        self.assertEqual(context.exception.errors()[0]["type"], "literal_error")


    def test_missing_scientific_name(self):
        data = {
            "species": {
                "common_name": "Barn Swallow"
                # Missing scientific_name
            },
            "confidence": "high",
            "visual_evidence": [
                "Dark blue-black upper body",
                "Rust-colored underparts",
                "Slender body shape"
            ],
            "alternative_species": [
                {
                    "common_name": "Cliff Swallow",
                    "scientific_name": "Petrochelidon pyrrhonota",
                    "reason": "Similar body shape and coloration"
                }
            ],
            "limitations": [
                "Tail markings are not clearly visible"
            ]
        }

        with self.assertRaises(ValidationError) as context:
            IdentificationResult.model_validate(data)

        self.assertEqual(context.exception.errors()[0]["type"], "missing")



    def test_invalid_visual_evidence_type(self):
        data = {
            "species": {
                "common_name": "Barn Swallow",
                "scientific_name": "Hirundo rustica"
            },
            "confidence": "high",
            "visual_evidence": [
                "Dark blue-black upper body",
                123,  # Invalid type: integer instead of string
                "Slender body shape"
            ],
            "alternative_species": [
                {
                    "common_name": "Cliff Swallow",
                    "scientific_name": "Petrochelidon pyrrhonota",
                    "reason": "Similar body shape and coloration"
                }
            ],
            "limitations": [
                "Tail markings are not clearly visible"
            ]
        }

        with self.assertRaises(ValidationError) as context:
            IdentificationResult.model_validate(data)

        self.assertEqual(context.exception.errors()[0]["type"], "string_type")
        self.assertEqual(context.exception.errors()[0]["loc"], ("visual_evidence", 1))


    def test_missing_limitations(self):
        data = {
            "species": {
                "common_name": "Barn Swallow",
                "scientific_name": "Hirundo rustica"
            },
            "confidence": "high",
            "visual_evidence": [
                "Dark blue-black upper body",
                "Rust-colored underparts",
                "Slender body shape"
            ],
            "alternative_species": [
                {
                    "common_name": "Cliff Swallow",
                    "scientific_name": "Petrochelidon pyrrhonota",
                    "reason": "Similar body shape and coloration"
                }
            ]
            # Missing limitations field
        }

        with self.assertRaises(ValidationError) as context:
            IdentificationResult.model_validate(data)

        self.assertEqual(context.exception.errors()[0]["type"], "missing")
        self.assertEqual(context.exception.errors()[0]["loc"], ("limitations",))


    def test_empty_lists_are_valid(self):
        data = {
            "species": {
                "common_name": "Barn Swallow",
                "scientific_name": "Hirundo rustica"
            },
            "confidence": "medium",
            "visual_evidence": [],
            "alternative_species": [],
            "limitations": []
        }

        result = IdentificationResult.model_validate(data)

        self.assertIsInstance(result, IdentificationResult)
        self.assertEqual(result.visual_evidence, [])
        self.assertEqual(result.alternative_species, [])
        self.assertEqual(result.limitations, [])


if __name__ == "__main__":
    unittest.main()