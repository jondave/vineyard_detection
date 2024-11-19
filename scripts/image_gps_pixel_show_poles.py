import subprocess
import json
from PIL import Image, ImageDraw
import math
import piexif

# Function to extract flight data (e.g., FlightYawDegree, GimbalYawDegree) and GPS coordinates from the image EXIF
def extract_flight_gimbal_degrees(image_path):
    try:
        # Run ExifTool with -json option
        result = subprocess.run(['exiftool', '-json', image_path], capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error: {result.stderr}")
            return None, None, None, None, None

        # Parse the JSON output
        metadata = json.loads(result.stdout)
        if metadata:
            metadata_dict = metadata[0]
            flight_degree = metadata_dict.get('FlightYawDegree')
            gimbal_degree = metadata_dict.get('GimbalYawDegree')
            gps_latitude = metadata_dict.get('GPSLatitude')
            gps_longitude = metadata_dict.get('GPSLongitude')
            gps_altitude = metadata_dict.get('GPSAltitude')

            # Extract numeric part from altitude if it contains "m Above Sea Level"
            if gps_altitude and isinstance(gps_altitude, str):
                altitude_value = gps_altitude.split()[0]  # Get the numeric part (before "m")
                gps_altitude = float(altitude_value)  # Convert to float

            return flight_degree, gimbal_degree, gps_latitude, gps_longitude, gps_altitude
        else:
            print("No metadata found.")
            return None, None, None, None, None
    except Exception as e:
        print(f"Error: {e}")
        return None, None, None, None, None

# Function to convert DMS (Degrees, Minutes, Seconds) to decimal degrees
def dms_to_decimal(degrees, minutes, seconds, direction):
    decimal = degrees + (minutes / 60) + (seconds / 3600)
    if direction in ['S', 'W']:
        decimal = -decimal
    return decimal

# Function to parse GPS DMS string into degrees, minutes, seconds, and direction
def parse_dms(dms_str):
    # Extract components from DMS string, which includes deg, ', ", and N/S/E/W
    dms_parts = dms_str.replace("deg", "").replace("'", "").replace('"', "").split()
    degrees = float(dms_parts[0])
    minutes = float(dms_parts[1])
    seconds = float(dms_parts[2])
    direction = dms_parts[3]
    
    return degrees, minutes, seconds, direction

# Function to extract the image's EXIF data
def extract_exif(image_path):
    img = Image.open(image_path)
    exif_data = piexif.load(img.info['exif'])  # Extract EXIF data from image
    return exif_data

# Function to extract GPS data from EXIF
def extract_gps(exif_data):
    gps_info = exif_data.get('GPS', {})
    if gps_info:
        latitude = gps_info.get(piexif.GPSIFD.GPSLatitude)
        latitude_ref = gps_info.get(piexif.GPSIFD.GPSLatitudeRef)
        longitude = gps_info.get(piexif.GPSIFD.GPSLongitude)
        longitude_ref = gps_info.get(piexif.GPSIFD.GPSLongitudeRef)

        if latitude and longitude and latitude_ref and longitude_ref:
            # Convert GPS coordinates from DMS to decimal degrees
            lat_deg, lat_min, lat_sec = latitude
            lon_deg, lon_min, lon_sec = longitude
            lat = dms_to_decimal(lat_deg[0], lat_min[0], lat_sec[0], latitude_ref.decode())
            lon = dms_to_decimal(lon_deg[0], lon_min[0], lon_sec[0], longitude_ref.decode())

            return lat, lon
    return None, None

# Function to calculate real-world distances based on pixel distances
def calculate_real_world_distance(pixel_distance, focal_length, fov_deg, image_width, sensor_width_mm=None):
    if sensor_width_mm is None:
        sensor_width_mm = 36  # Assuming full-frame sensor (36mm)
    
    fov_rad = math.radians(fov_deg)
    image_width_mm = sensor_width_mm * (image_width / 4056)  # Scale sensor width to match image width
    real_world_distance_per_pixel = (math.tan(fov_rad / 2) * 2 * image_width_mm) / image_width
    
    real_world_distance = pixel_distance * real_world_distance_per_pixel
    return real_world_distance

# Function to get new latitude and longitude for a pixel
def get_gps_from_pixel(x, y, flight_degree, gimbal_degree, image_width, image_height, real_world_distance_per_pixel, latitude, longitude):
    # Calculate pixel displacement from the center of the image
    pixel_x = x - (image_width / 2)
    pixel_y = y - (image_height / 2)

    # Correct the pixel distances based on flight and gimbal orientation
    corrected_pixel_x = pixel_x * math.cos(math.radians(flight_degree))  # Adjusting by flight yaw
    corrected_pixel_y = pixel_y * math.cos(math.radians(gimbal_degree))  # Adjusting by gimbal pitch

    # Calculate the real-world distance
    lat_change = corrected_pixel_y * real_world_distance_per_pixel
    lon_change = corrected_pixel_x * real_world_distance_per_pixel

    # Convert real-world changes to geographic changes in lat/lon
    new_latitude = latitude + (lat_change / 111320)  # Approximate conversion for latitude
    new_longitude = longitude + (lon_change / (40008000 * math.cos(math.radians(latitude))))  # Approx for longitude

    return new_latitude, new_longitude

def get_pixel_from_gps(latitude, longitude, flight_degree, gimbal_degree, image_width, image_height, real_world_distance_per_pixel, gps_lat_decimal, gps_lon_decimal):
    # Calculate the real-world displacement in meters from the given lat/lon to the center point
    lat_change = (latitude - gps_lat_decimal) * 111320  # Approximate conversion to meters for latitude
    lon_change = (longitude - gps_lon_decimal) * (40008000 * math.cos(math.radians(latitude)) / 360)  # Conversion to meters for longitude

    # Convert flight orientation to radians
    flight_radians = math.radians(flight_degree)

    # Adjust the displacement for flight orientation
    corrected_lon_change = lon_change * math.cos(flight_radians) - lat_change * math.sin(flight_radians)
    corrected_lat_change = lon_change * math.sin(flight_radians) + lat_change * math.cos(flight_radians)

    # Adjust pixel distance using real-world distance per pixel
    corrected_pixel_x = corrected_lon_change / real_world_distance_per_pixel
    corrected_pixel_y = corrected_lat_change / real_world_distance_per_pixel

    # Convert to pixel coordinates (adjusting for the center of the image)
    pixel_x = (image_width / 2) + corrected_pixel_x
    pixel_y = (image_height / 2) - corrected_pixel_y  # Invert Y-axis for typical image coordinates

    return int(pixel_x), int(pixel_y)

# Function to draw circles on the image based on lat/lon coordinates
def draw_circles_on_image(image_path, gps_points, flight_degree, gimbal_degree):
    # Open the image
    img = Image.open(image_path)
    image_width, image_height = img.size
    draw = ImageDraw.Draw(img)

    # Camera specifications
    focal_length_mm = 4.5
    fov_deg = 73.7
    sensor_width_mm = 6.3

    # Get the center GPS coordinates (including altitude)
    flight_degree, gimbal_degree, gps_latitude, gps_longitude, gps_altitude = extract_flight_gimbal_degrees(image_path)

    flight_degree = float(flight_degree) if flight_degree else 0.0
    gimbal_degree = float(gimbal_degree) if gimbal_degree else 0.0
    
    gps_lat_deg, gps_lat_min, gps_lat_sec, lat_ref = parse_dms(gps_latitude)
    gps_lat_decimal = dms_to_decimal(gps_lat_deg, gps_lat_min, gps_lat_sec, lat_ref)

    gps_lon_deg, gps_lon_min, gps_lon_sec, lon_ref = parse_dms(gps_longitude)
    gps_lon_decimal = dms_to_decimal(gps_lon_deg, gps_lon_min, gps_lon_sec, lon_ref)

    # Altitude correction (assuming the altitude is in meters)
    altitude_meters = float(gps_altitude)

    # Calculate the real-world distance per pixel (adjusted for altitude)
    fov_rad = math.radians(fov_deg)
    # Adjusted field of view based on altitude (simplified approach)
    adjusted_fov_deg = fov_deg * (1 + altitude_meters / 1000)  # Simple correction for altitude
    adjusted_fov_rad = math.radians(adjusted_fov_deg)
    
    real_world_distance_per_pixel = (math.tan(adjusted_fov_rad / 2) * 2 * sensor_width_mm) / image_width

    for lat, lon in gps_points:
        # Convert lat/lon to pixel coordinates
        pixel_x, pixel_y = get_pixel_from_gps(lat, lon, flight_degree, gimbal_degree, image_width, image_height, real_world_distance_per_pixel, gps_lat_decimal, gps_lon_decimal)
        
        # Draw a circle at the calculated pixel coordinates
        radius = 100  # Radius of the circle in pixels
        draw.ellipse([pixel_x - radius, pixel_y - radius, pixel_x + radius, pixel_y + radius], outline="red", width=20)
        print("Circle drawn at:", pixel_x, pixel_y)

    # Save the modified image with circles
    output_image_path = "../images/output_image_with_circles.jpg"
    img.save(output_image_path)

# Example of using pixel_x and pixel_y in the main function
def process_image(image_path, gps_points):
    flight_degree, gimbal_degree, gps_latitude, gps_longitude, gps_altitude = extract_flight_gimbal_degrees(image_path)

    flight_degree = float(flight_degree) if flight_degree else 0.0
    gimbal_degree = float(gimbal_degree) if gimbal_degree else 0.0

    gps_lat_deg, gps_lat_min, gps_lat_sec, lat_ref = parse_dms(gps_latitude)
    gps_lat_decimal = dms_to_decimal(gps_lat_deg, gps_lat_min, gps_lat_sec, lat_ref)

    gps_lon_deg, gps_lon_min, gps_lon_sec, lon_ref = parse_dms(gps_longitude)
    gps_lon_decimal = dms_to_decimal(gps_lon_deg, gps_lon_min, gps_lon_sec, lon_ref)
    
    if flight_degree:
        print(f"Flight Degree: {flight_degree}")
    if gimbal_degree:
        print(f"Gimbal Degree: {gimbal_degree}")
    if gps_latitude and gps_longitude:
        print(f"GPS Coordinates: Latitude = {gps_lat_decimal}, Longitude = {gps_lon_decimal}")

    draw_circles_on_image(image_path, gps_points, flight_degree, gimbal_degree)

if __name__ == "__main__":
    image_path = "../images/39_feet/DJI_20240802143249_0121_W.JPG"
    gps_points = [
            (53.26818842,-0.52427737),
            (53.26813837,-0.52426541),
            (53.26808856,-0.52425335),
            (53.26803849,-0.52424047),
            (53.26818522,-0.52431449),
            (53.26813509,-0.52430208),
            (53.26808532,-0.52428968),
            (53.26803515,-0.52427742),
            (53.26818187,-0.52435181),
            (53.26813158,-0.52433952),
            (53.26808211,-0.52432693),
            (53.26803182,-0.52431475),
            (53.26817882,-0.52438866),
            (53.26812848,-0.52437636),
            (53.26807903,-0.52436409),
            (53.26802873,-0.52435185),
            (53.26817541,-0.52442589),
            (53.26812517,-0.5244135),
            (53.26807555,-0.5244011),
            (53.2680255,-0.52438878),
            (53.26817238,-0.52446323),
            (53.26812194,-0.52445077),
            (53.26807253,-0.52443819),
            (53.26802228,-0.52442619),
            (53.26816928,-0.52449965),
            (53.26811864,-0.52448766),
            (53.26806932,-0.52447579),
            (53.26801926,-0.52446331),
            (53.26816599,-0.52453691),
            (53.26811528,-0.5245244),
            (53.26806603,-0.52451219),
            (53.26801591,-0.52449999),
            (53.26816264,-0.52457417),
            (53.2681122,-0.52456217),
            (53.26806294,-0.52454963),
            (53.26801275,-0.52453719),
            (53.26815947,-0.52461139),
            (53.26810906,-0.52459885),
            (53.26805976,-0.52458653),
            (53.26800959,-0.52457471)#,

            #(53.26815, -0.524575), # centre of image to check if it is correct
            #(53.268175184088804, -0.5245453595031128) # random point
    ]
    process_image(image_path, gps_points)