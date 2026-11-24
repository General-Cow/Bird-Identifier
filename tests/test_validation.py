from app.validation import IdentificationResult, validate_semantics
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


    def test_high_confidence_without_evidence(self):
        data = {
            "species": {
                "common_name": "Barn Swallow",
                "scientific_name": "Hirundo rustica"
            },
            "confidence": "high",
            "visual_evidence": [],
            "alternative_species": [],
            "limitations": []
        }

        result = IdentificationResult.model_validate(data)

        with self.assertRaises(ValueError) as context:
            validate_semantics(result)

        self.assertEqual(str(context.exception), "High confidence results must have visual evidence.")


    def test_high_confidence_with_evidence(self):
        data = {
            "species": {
                "common_name": "Barn Swallow",
                "scientific_name": "Hirundo rustica"
            },
            "confidence": "high",
            "visual_evidence": ["Dark blue-black upper body"],
            "alternative_species": [],
            "limitations": []
        }

        result = IdentificationResult.model_validate(data)

        # This should not raise an exception
        validate_semantics(result)


    def test_no_species_high_confidence(self):
        data = {
            "species": None,
            "confidence": "high",
            "visual_evidence": ["Dark blue-black upper body"],
            "alternative_species": [],
            "limitations": []
        }

        result = IdentificationResult.model_validate(data)

        with self.assertRaises(ValueError) as context:
            validate_semantics(result)

        self.assertEqual(str(context.exception), "High confidence results must have a species identified.")


    def test_no_species_low_confidence(self):
        data = {
            "species": None,
            "confidence": "low",
            "visual_evidence": [],
            "alternative_species": [],
            "limitations": []
        }

        result = IdentificationResult.model_validate(data)

        # This should not raise an exception
        validate_semantics(result)

    def test_alternative_species_same_as_identified(self):
        data = {
            "species": {
                "common_name": "Barn Swallow",
                "scientific_name": "Hirundo rustica"
            },
            "confidence": "medium",
            "visual_evidence": ["Dark blue-black upper body"],
            "alternative_species": [
                {
                    "common_name": "Barn Swallow",
                    "scientific_name": "Hirundo rustica",  # Same as identified species
                    "reason": "Similar appearance"
                }
            ],
            "limitations": []
        }

        result = IdentificationResult.model_validate(data)

        with self.assertRaises(ValueError) as context:
            validate_semantics(result)

        self.assertEqual(str(context.exception), "Alternative species cannot have the same scientific name as the identified species.")


    def test_alternative_species_different_from_identified(self):
        data = {
            "species": {
                "common_name": "Barn Swallow",
                "scientific_name": "Hirundo rustica"
            },
            "confidence": "medium",
            "visual_evidence": ["Dark blue-black upper body"],
            "alternative_species": [
                {
                    "common_name": "Cliff Swallow",
                    "scientific_name": "Petrochelidon pyrrhonota",  # Different from identified species
                    "reason": "Similar appearance"
                }
            ],
            "limitations": []
        }

        result = IdentificationResult.model_validate(data)

        # This should not raise an exception
        validate_semantics(result)


if __name__ == "__main__":
    unittest.main()