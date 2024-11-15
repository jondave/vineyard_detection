from PIL import Image
import exifread

def normalize_angle(angle):
    """
    Normalize an angle to the range [0, 360).
    """
    return angle % 360

def get_decimal_coordinates(coords, ref):
    """
    Convert GPS coordinates to decimal degrees.
    """
    degrees = coords[0]
    minutes = coords[1]
    seconds = coords[2]
    decimal = degrees + (minutes / 60.0) + (seconds / 3600.0)
    if ref in ["S", "W"]:
        decimal = -decimal
    return decimal

def get_image_metadata(image_path):
    """
    Extract GPS coordinates, orientation, and compass heading from an image.
    """
    with open(image_path, 'rb') as img_file:
        tags = exifread.process_file(img_file, details=False)

    # Extract GPS coordinates
    gps_latitude = tags.get("GPS GPSLatitude")
    gps_latitude_ref = tags.get("GPS GPSLatitudeRef")
    gps_longitude = tags.get("GPS GPSLongitude")
    gps_longitude_ref = tags.get("GPS GPSLongitudeRef")
    gps_img_direction = tags.get("GPS GPSImgDirection")  # Compass heading
    gps_img_direction_ref = tags.get("GPS GPSImgDirectionRef")
    orientation = tags.get("Image Orientation")

    if gps_latitude and gps_latitude_ref and gps_longitude and gps_longitude_ref:
        lat_decimal = get_decimal_coordinates(gps_latitude.values, gps_latitude_ref.values)
        lon_decimal = get_decimal_coordinates(gps_longitude.values, gps_longitude_ref.values)
    else:
        lat_decimal = None
        lon_decimal = None

    # Extract compass heading
    if gps_img_direction:
        compass_heading = float(gps_img_direction.values[0])
        compass_ref = gps_img_direction_ref.values if gps_img_direction_ref else "True North"
    else:
        compass_heading = None
        compass_ref = None

    # Extract orientation
    orientation_mapping = {
        1: "Top is North",
        3: "Top is South",
        6: "Top is East",
        8: "Top is West",
    }
    orientation_description = orientation_mapping.get(orientation.values[0], "Unknown") if orientation else "Unknown"

    # Extract GimbalDegree and FlightDegree (custom logic)
    gimbal_degree = tags.get("GimbalDegree")
    flight_degree = tags.get("FlightDegree")

    print("gimbal_degree: ", gimbal_degree)
    print("flight_degree: ", flight_degree)

    gimbal_heading = normalize_angle(float(gimbal_degree.values[0])) if gimbal_degree else None
    flight_heading = normalize_angle(float(flight_degree.values[0])) if flight_degree else None

    return {
        "latitude": lat_decimal,
        "longitude": lon_decimal,
        "orientation": orientation_description,
        "compass_heading": compass_heading,
        "compass_ref": compass_ref,
        "gimbal_heading": gimbal_heading,
        "flight_heading": flight_heading,
    }

# Example usage
image_path = "../images/39_feet/DJI_20240802142835_0003_W.JPG"  # Replace with your image path
# metadata = get_image_metadata(image_path)

# print(f"GPS Location: Latitude {metadata['latitude']}, Longitude {metadata['longitude']}")
# print(f"Orientation: {metadata['orientation']}")
# if metadata['compass_heading'] is not None:
#     print(f"Compass Heading: {metadata['compass_heading']}° ({metadata['compass_ref']})")
# else:
#     print("Compass Heading: Not available")
# if metadata['gimbal_heading'] is not None:
#     print(f"Gimbal Compass Heading: {metadata['gimbal_heading']}°")
# else:
#     print("Gimbal Compass Heading: Not available")
# if metadata['flight_heading'] is not None:
#     print(f"Flight Compass Heading: {metadata['flight_heading']}°")
# else:
#     print("Flight Compass Heading: Not available")

with open(image_path, 'rb') as img_file:
    tags = exifread.process_file(img_file, details=False)
for tag in tags:
    print(f"{tag}: {tags[tag]}")
