from pathlib import Path


custom_image_path = Path(__file__).parent.parent / "data" / "04-pizza-dad.jpeg"

print(custom_image_path.is_file())