import json
from pydantic import BaseModel, ValidationError
from typing import Literal


def parse_json(json_string):
    """
    Parses a JSON string into a Python object.

    Args:
        json_string (str): The JSON string to parse.

    Returns:
        object: The parsed JSON value, which could be a dictionary, list, string, number, boolean, or None

    Raises:
        TypeError: If the input is not a string.
        ValueError: If the provided string is not valid JSON.

    """
    if not isinstance(json_string, str):
        raise TypeError("Input must be a string.")

    try:
        json_data = json.loads(json_string)
        return json_data
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON at position {e.pos}: {e.msg}") from e

class Species(BaseModel):
    common_name: str
    scientific_name: str

class AlternativeSpecies(BaseModel):
    common_name: str
    scientific_name: str
    reason: str

class IdentificationResult(BaseModel):
    """
    A Pydantic model to validate the structure of the identification result JSON data.
    """
    species: Species | None
    confidence: Literal["high", "medium", "low"]
    visual_evidence: list[str]
    alternative_species: list[AlternativeSpecies]
    limitations: list[str]


def validate_semantics(result: IdentificationResult):
    """Validates the semantics of the identification result."""
    if not result.visual_evidence and result.confidence == "high":
        raise ValueError("High confidence results must have visual evidence.")
    if result.species is None and result.confidence == "high":
        raise ValueError("High confidence results must have a species identified.")
    if result.species is not None:
        for alt_species in result.alternative_species:
            if alt_species.scientific_name == result.species.scientific_name:
                raise ValueError("Alternative species cannot have the same scientific name as the identified species.")