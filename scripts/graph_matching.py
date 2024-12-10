import json
import numpy as np
from scipy.spatial import distance
import math

# Load the GeoJSON file
with open('../data/clustered_poles.geojson', 'r') as f:
    geojson_data = json.load(f)

# Extract coordinates
coordinates = np.array([feature['geometry']['coordinates'] for feature in geojson_data['features']])

# Define prior knowledge
poles_per_row = [3, 4, 4, 4, 4, 4, 4, 4, 4, 4]  # poles per row
compass_heading = 350 # compass heading of rows

# Convert to mathematical heading (counter-clockwise from 0 deg North)
math_heading = (360 - compass_heading) % 360

# Convert heading to radians
heading_radians = math.radians(math_heading)

# Unit vector for the heading direction
direction_vector = np.array([math.cos(heading_radians), math.sin(heading_radians)])

# Project coordinates onto the heading direction
# coordinates: array of [longitude, latitude]
projections = np.dot(coordinates, direction_vector)

# Sort coordinates by their projections
coordinates_sorted = coordinates[np.argsort(projections)]

# Sort poles by latitude to get an initial rough alignment (or longitude, depending on orientation)
# coordinates_sorted = coordinates[coordinates[:, 1].argsort()]  # Sort by latitude
# coordinates_sorted = coordinates[coordinates[:, 0].argsort()]  # Sort by longitude

# Divide sorted coordinates into rows
row_assignments = []
start_idx = 0

for row_idx, count in enumerate(poles_per_row):
    end_idx = start_idx + count
    row_assignments.append({
        "row_id": row_idx + 1,
        "coordinates": coordinates_sorted[start_idx:end_idx]
    })
    start_idx = end_idx

# Create GeoJSON features for rows as LineStrings
row_features = []
for row in row_assignments:
    # Sort within each row to ensure consistent LineString geometry
    row['coordinates'] = sorted(row['coordinates'].tolist(), key=lambda x: x[0])  # Sort by longitude
    row_features.append({
        "type": "Feature",
        "geometry": {
            "type": "LineString",
            "coordinates": row['coordinates']
        },
        "properties": {
            "row_id": row['row_id']
        }
    })

# Add row_id to individual poles
pole_features = []
for feature, coord in zip(geojson_data['features'], coordinates_sorted):
    for row in row_assignments:
        if coord.tolist() in row['coordinates']:
            feature['properties']['row_id'] = row['row_id']
            pole_features.append(feature)
            break

# Combine both poles and rows into a single GeoJSON
updated_geojson = {
    "type": "FeatureCollection",
    "features": pole_features + row_features
}

# Save the updated GeoJSON
with open('../data/updated_poles_and_rows.geojson', 'w') as f:
    json.dump(updated_geojson, f, indent=2)

print("GeoJSON with prior knowledge saved as 'updated_poles_and_rows.geojson'")
