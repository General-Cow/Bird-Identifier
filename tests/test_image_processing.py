from app.image_processing import ImageProcessor


def main():
    processor = ImageProcessor(
        max_dimension=2048,
        jpeg_quality=90,
    )

    result = processor.process(
        image_path="data/input/IMG_0374.jpg",
        output_dir="data/processed",
    )

    print("\n" + "=" * 50)
    print("IMAGE PROCESSING RESULTS")
    print("=" * 50)

    print(f"Original file:    {result.original_path}")
    print(f"Processed file:   {result.processed_path}")
    print(f"Original format:  {result.original_format}")
    print(f"Original size:    {result.original_size}")
    print(f"Processed size:   {result.processed_size}")
    print(f"Date taken:       {result.date_taken}")

    print("\nProcessing complete!")


if __name__ == "__main__":
    main()