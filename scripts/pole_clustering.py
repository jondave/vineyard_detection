import json
from sklearn.cluster import DBSCAN
import numpy as np
from shapely.geometry import Point, MultiPoint

# Input and output GeoJSON file paths
input_geojson_file = "../data/detected_pole_coordinates.geojson"
output_geojson_file = "../data/clustered_poles.geojson"

# Load the input GeoJSON file
with open(input_geojson_file, "r") as file:
    geojson_data = json.load(file)

# Extract coordinates from the GeoJSON file
coordinates = [
    feature["geometry"]["coordinates"]
    for feature in geojson_data["features"]
    if feature["geometry"]["type"] == "Point"
]

# Convert to NumPy array
coords = np.array(coordinates)

# Run DBSCAN clustering
dbscan = DBSCAN(eps=0.0000002, min_samples=3, metric="haversine")  # Adjust `eps` and `min_samples` as needed
coords_radians = np.radians(coords)
labels = dbscan.fit_predict(coords_radians)

# Create a new GeoJSON FeatureCollection for the clustered centroids
clustered_geojson = {
    "type": "FeatureCollection",
    "features": []
}

# Process each cluster
unique_labels = set(labels)
for cluster_id in unique_labels:
    if cluster_id == -1:
        # Skip noise points (label -1)
        continue

    # Extract points belonging to this cluster
    cluster_points = coords[labels == cluster_id]

    # Compute the centroid of the cluster
    cluster_geometry = MultiPoint([Point(p[0], p[1]) for p in cluster_points])
    cluster_centroid = cluster_geometry.centroid

    # Append the centroid to the GeoJSON
    feature = {
        "type": "Feature",
        "geometry": {
            "type": "Point",
            "coordinates": [cluster_centroid.x, cluster_centroid.y]
        },
        "properties": {
            "type": "pole",
            "pole_id": int(cluster_id)  # Optional: Include the pole / cluster ID
        }
    }
    clustered_geojson["features"].append(feature)

# Save the clustered GeoJSON to a file
with open(output_geojson_file, "w") as file:
    json.dump(clustered_geojson, file, indent=4)

print(f"Clustered GeoJSON saved to: {output_geojson_file}")
