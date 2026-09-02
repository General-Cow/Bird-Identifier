from src.BirdVLM import BirdVLM
from pathlib import Path
from PIL import Image

def main():
    # Example usage of the BirdVLM class
    bird_identifier = BirdVLM()

    # Load an image (replace 'path_to_image.jpg' with your actual image path)
    image_path = Path(r'data\IMG_1713.jpg')
    image = Image.open(image_path)
    
    # Define location, date, and habitat
    location = "Virginia, USA"
    date = "30 August"
    habitat = "By the Potomac River in a forested area with a nearby marsh"
    
    # Identify the bird in the image
    result = bird_identifier.identify_bird(image, location, date, habitat)
    
    # Print the result
    print(result)

if __name__ == "__main__":
    main()