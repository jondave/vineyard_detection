import struct
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
import string

def clean_value(value):
    """Remove non-printable characters for readability."""
    if isinstance(value, bytes):
        # Decode bytes to string, ignoring errors
        value = value.decode("utf-8", errors="ignore")
    if isinstance(value, str):
        # Keep only printable characters
        return ''.join(filter(lambda x: x in string.printable, value))
    return value

def print_raw_data(maker_note_bytes):
    """Print raw MakerNote data in hexadecimal format for inspection."""
    hex_data = maker_note_bytes.hex()  # Convert bytes to hex string
    # Print the first 500 hex characters to inspect the structure
    print(f"Raw MakerNote Data (Hex): {hex_data[:500]}...")

def extract_flight_and_gimbal_degrees(maker_note_bytes):
    """Try to extract flight and gimbal degrees from raw MakerNote data."""
    try:
        # Print raw hex data for inspection
        print_raw_data(maker_note_bytes)

        # Example of interpreting bytes as signed 16-bit integers (shorts)
        flight_degree = struct.unpack_from('<h', maker_note_bytes, offset=10)[0]  # Little-endian 16-bit int
        gimbal_degree = struct.unpack_from('<h', maker_note_bytes, offset=12)[0]  # Little-endian 16-bit int

        print(f"Extracted Flight Degree: {flight_degree}")
        print(f"Extracted Gimbal Degree: {gimbal_degree}")
        
    except Exception as e:
        print(f"Error extracting flight and gimbal degrees: {e}")

def extract_exif_data(image_path):
    try:
        # Open the image file
        with Image.open(image_path) as img:
            exif_data = img._getexif()  # Extract EXIF data
            if not exif_data:
                print("No EXIF data found.")
                return

            # Map EXIF tags to their human-readable names
            for tag_id, value in exif_data.items():
                tag = TAGS.get(tag_id, tag_id)  # Get the tag name or default to the ID
                
                # Clean and format the value
                clean_val = clean_value(value)
                # print(f"{tag}: {clean_val}")

                # Special handling for GPSInfo
                if tag == "GPSInfo":
                    print("\nGPS Metadata:")
                    for gps_id, gps_value in value.items():
                        gps_tag = GPSTAGS.get(gps_id, gps_id)
                        clean_gps_val = clean_value(gps_value)
                        print(f"  {gps_tag}: {clean_gps_val}")

                # Special handling for MakerNote
                if tag == "MakerNote":
                    print("\nMakerNote Metadata:")
                    if isinstance(value, bytes):
                        # Print MakerNote as raw bytes or try to decode
                        print(f"  Raw MakerNote data (bytes): {value[:100]}...")  # Print the first 100 bytes as a preview
                        extract_flight_and_gimbal_degrees(value)  # Attempt to extract degrees from raw data
                    else:
                        # If it's not bytes, assume it's some other structure that needs special handling
                        print(f"  MakerNote data: {clean_value(value)}")

    except FileNotFoundError:
        print(f"File not found: {image_path}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    # Replace 'your_image.jpg' with the path to your JPEG image
    image_path = "../images/39_feet/DJI_20240802142835_0003_W.JPG"
    print(f"Extracting metadata from {image_path}...\n")
    extract_exif_data(image_path)
