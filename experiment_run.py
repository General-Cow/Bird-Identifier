from src.BirdVLM import BirdVLM
from pathlib import Path
from PIL import Image
import pandas as pd
import tqdm

def main():
    # Example usage of the BirdVLM class
    bird_identifier = BirdVLM()

    # Collect lists from text files for experimental run
    path_to_images = Path(r"C:\Users\dblan\Desktop\Personal Python\Birds\nabirds")

    with open(path_to_images / "images.txt", "r") as f:
        image_line = [line.strip().split() for line in f.readlines()]
        # image_ids = [line[0] for line in image_line]
        # image_paths = [line[1] for line in image_line]

    with open(path_to_images / "image_class_labels.txt", "r") as f:
        class_labels = [line.strip().split() for line in f.readlines()]

    with open(path_to_images / "classes.txt", "r") as f:
        class_keys = [line.strip().split(maxsplit=1) for line in f.readlines()]

    # Create DataFrames for easier manipulation
    df_ids = pd.DataFrame(image_line, columns=["image_id", "image_path"])
    df_class_labels = pd.DataFrame(class_labels, columns=["image_id", "class_label"])
    df_class_keys = pd.DataFrame(class_keys, columns=["class_label", "class_name"])
    df = df_ids.merge(df_class_labels, on="image_id").merge(df_class_keys, on="class_label")
    df["result"] = ""

    # Define location, date, and habitat
    location = "North America"
    date = "unknown"
    habitat = "unknown"

    # Loop and run the bird identification for the available images, up to 100
    for i in tqdm.tqdm(range(min(100, len(df))), desc="Processing Images"):
        # Load an image
        image_path = Path(rf"C:\Users\dblan\Desktop\Personal Python\Birds\nabirds\images\{df.iloc[i]['image_path']}")
        image = Image.open(image_path)

        # Identify the bird in the image
        result = bird_identifier.identify_bird(image, location, date, habitat)

        # Save the result back to the matching row only
        df.loc[i, "result"] = result["species"]

    # Save the results to a CSV file
    df.to_csv("./bird_identification_results.csv", index=False)

if __name__ == "__main__":
    main()