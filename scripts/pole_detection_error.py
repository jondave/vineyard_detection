import json
from geopy.distance import geodesic

# Load detected pole coordinates (YOLO detected poles)
with open('../data/clustered_poles.geojson') as f:
    detected_poles = json.load(f)

# Load RTK GPS pole coordinates
with open('../data/rtk_gps_poles.geojson') as f:
    rtk_poles = json.load(f)

# Function to calculate the closest RTK pole for each detected pole
def calculate_closest_pole_error(detected_poles, rtk_poles):
    errors = []
    
    # Iterate through each detected pole
    for detected_pole in detected_poles['features']:
        detected_coords = detected_pole['geometry']['coordinates']
        
        # Find the closest RTK pole
        closest_error = float('inf')  # Set an initial high error
        closest_pole = None
        
        for rtk_pole in rtk_poles['features']:
            rtk_coords = rtk_pole['geometry']['coordinates']
            # Calculate the distance between detected pole and RTK pole
            distance = geodesic(detected_coords[::-1], rtk_coords[::-1]).meters  # Reverse coordinates for geopy (lat, lon)
            
            # Update closest pole if this one is closer
            if distance < closest_error:
                closest_error = distance
                closest_pole = rtk_pole
        
        # Store the closest pole and the error
        errors.append({
            'detected_pole': detected_pole['geometry']['coordinates'],
            'closest_pole': closest_pole['geometry']['coordinates'],
            'error_in_meters': closest_error
        })
    
    return errors

# Call the function and get the error data
errors = calculate_closest_pole_error(detected_poles, rtk_poles)

# Output the results and calculate the average error
total_error = 0
for error in errors:
    print(f"Detected pole at {error['detected_pole']} has an error of {error['error_in_meters']} meters.")
    print(f"Closest RTK pole is at {error['closest_pole']}")
    print("-" * 50)
    total_error += error['error_in_meters']

# Calculate and print the average error
average_error = total_error / len(errors) if errors else 0
print(f"Average error: {average_error} meters")
print(f"Number of RTK GPS poles: {len(rtk_poles['features'])}")
print(f"Number of Detected poles: {len(detected_poles['features'])}")
