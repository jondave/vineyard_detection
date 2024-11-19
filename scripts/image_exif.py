import subprocess
import json

def extract_exif_data(image_path):
    try:
        # Call ExifTool to extract metadata in JSON format
        result = subprocess.run(['exiftool', '-json', image_path], capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error: {result.stderr}")
            return

        # Parse the JSON output from ExifTool
        metadata = json.loads(result.stdout)
        if metadata:
            for tag, value in metadata[0].items():
                print(f"{tag}: {value}")
        else:
            print("No metadata found.")
    except Exception as e:
        print(f"Error: {e}")

# Replace with the path to your image
image_path = "../images/39_feet/DJI_20240802142835_0003_W.JPG"
extract_exif_data(image_path)
