import subprocess
import json

def extract_flight_gimbal_degrees(image_path):
    try:
        # Run ExifTool with -json option
        result = subprocess.run(['exiftool', '-json', image_path], capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error: {result.stderr}")
            return

        # Parse the JSON output
        metadata = json.loads(result.stdout)
        if metadata:
            # Look for the specific fields in the metadata
            metadata_dict = metadata[0]
            flight_degree = metadata_dict.get('FlightYawDegree')
            gimbal_degree = metadata_dict.get('GimbalYawDegree')
            gps_latitude = metadata_dict.get('GPSLatitude')
            gps_longitude = metadata_dict.get('GPSLongitude')
            gpr_altitude = metadata_dict.get('GPSAltitude')

            if flight_degree is not None:
                print(f"Flight Degree: {flight_degree}")
            else:
                print("Flight Degree not found.")

            if gimbal_degree is not None:
                print(f"Gimbal Degree: {gimbal_degree}")
            else:
                print("Gimbal Degree not found.")

            if gps_latitude is not None:
                print(f"GPS Latitude: {gps_latitude}")
            else:
                print("GPS Latitude not found.")

            if gps_longitude is not None:
                print(f"GPS Longitude: {gps_longitude}")
            else:
                print("GPS Longitude not found.")

            if gpr_altitude is not None:
                print(f"GPS Altitude: {gpr_altitude}")
            else:
                print("GPS Altitude not found.")
        else:
            print("No metadata found.")
    except Exception as e:
        print(f"Error: {e}")

# Replace with the path to your image
image_path = "../images/39_feet/DJI_20240802142835_0003_W.JPG"
extract_flight_gimbal_degrees(image_path)
