VLM-based tool for identifying birds species.

Things to add:
- Model selection
- image processing with options
- location and other data (time of year) info entry to aid in identification
- cleaner output of results. Possibly a file with ref to image.
- qualititive confidence from model?
- Reasoning output?
- may want different output types. such as a simple output that just states species and a longer output that adds reasoning.
- consider output file type/types
- dont forget to modify stuff like output limit and such

Notes
- A test where I added a year to the model caused it to think a previous date was in the future presumably related to when it was trained. will likely want to drop year as itll just confuse the model and is somewhat irrelevant to identification.
- An example with two Canada Geese returned a response recognizing there was more than one bird in the image. More testing on multiple types of birds is needed. Long term may want to incorporate logic or something to seperate the birds.

Current example response

{
    "species": "Great Blue Heron",
    "scientific_name": "Ardea herodias",
    "confidence": "high",
    "reasoning": "The bird exhibits the characteristic long neck, legs, and sharp beak of a Great Blue Heron, commonly found in Virginia lakes during May. Its gray-blue plumage and posture are consistent with this species, which is a resident or migratory bird in the region during this time of year.",
    "alternatives": [
        {
            "species": "Snowy Egret",
            "reason": "Snowy Egrets are smaller, have white plumage, and typically have a more slender build; they are less likely to be seen in Virginia lakes in May without specific habitat context."
        },
        {
            "species": "Green Heron",
            "reason": "Green Herons are smaller and have a more compact body with a shorter neck; they are more common in marshy areas and may not be as frequently seen in open lake habitats."
        }
    ]
}