from src.BirdVLM import BirdVLM
from pathlib import Path
from PIL import Image

def main():
    # Example usage of the BirdVLM class
    bird_identifier = BirdVLM()

    # Load an image (replace 'path_to_image.jpg' with your actual image path)
    image_path = Path('path_to_image.jpg')
    image = Image.open(image_path)
    
    # Define location, date, and habitat
    location = "Virginia, USA"
    date = "11 May"
    habitat = "On a lake boardwalk, surrounded by a forest."
    
    # Identify the bird in the image
    result = bird_identifier.identify_bird(image, location, date, habitat)
    
    # Print the result
    print(result)

if __name__ == "__main__":
    main()