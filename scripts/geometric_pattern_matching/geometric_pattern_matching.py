import geopandas as gpd
import numpy as np
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN
import json
import matplotlib.pyplot as plt
from pyproj import CRS, Transformer

# Vineyard row parameters (now variable)
pole_spacing = 5.65  # meters
row_spacing = 2.75  # meters
num_poles_per_row = [4, 4, 4, 4, 4, 4, 4, 4, 4, 4]  # Example

# Load pole locations from GeoJSON
geojson_file = "../../data/clustered_poles.geojson"
gdf = gpd.read_file(geojson_file)

# Define the target CRS
target_crs = CRS.from_string("EPSG:4326")

# Transform to the target CRS
gdf_projected = gdf.to_crs(target_crs)

# Now get the coordinates in the projected CRS
pole_locations = np.array(list(zip(gdf_projected.geometry.x, gdf_projected.geometry.y)))


# # Preprocessing: Remove duplicate or very close points (if any)
# unique_mask = np.ones(len(pole_locations), dtype=bool)
# for i in range(len(pole_locations)):
#     for j in range(i + 1, len(pole_locations)):
#         if np.linalg.norm(pole_locations[i] - pole_locations[j]) < 0.5:  # Example threshold: 0.5m
#             unique_mask[j] = False
#             print("removing pole")
# pole_locations = pole_locations[unique_mask]

# KD-Tree for nearest neighbor search
tree = cKDTree(pole_locations)

def find_rows(seed_pole, row_size):
    row_poles = [seed_pole]
    for direction in [-1, 1]:  # Extend in both directions
        current_pole = seed_pole
        poles_added = 0  # Keep track of how many poles we've added in this direction
        while poles_added < row_size - 1:  # Changed to a while loop
            distances, indices = tree.query(current_pole, k=15, distance_upper_bound=pole_spacing * 2)  # Increased search radius
            neighbors = pole_locations[indices[distances < pole_spacing * 2]] # increased the search radius

            if len(neighbors) == 0:
                break  # No neighbors found, but don't give up entirely

            best_neighbor = None
            min_diff = float('inf')
            for neighbor in neighbors:
                diff = abs(np.linalg.norm(neighbor - current_pole) - pole_spacing)
                if diff < min_diff:
                    min_diff = diff
                    best_neighbor = neighbor

            if best_neighbor is not None:
                row_poles.append(best_neighbor) if direction == 1 else row_poles.insert(0, best_neighbor)
                current_pole = best_neighbor
                poles_added += 1
            else:
                # Handle missing poles: Skip ahead by pole_spacing
                predicted_next_pole = current_pole + (current_pole - row_poles[-2 if direction == 1 else 1]) / np.linalg.norm(current_pole - row_poles[-2 if direction == 1 else 1]) * pole_spacing if len(row_poles) > 1 else current_pole + np.array([pole_spacing,0]) # predict the next pole based on the vector between the last two poles
                row_poles.append(predicted_next_pole) if direction == 1 else row_poles.insert(0,predicted_next_pole)
                current_pole = predicted_next_pole
                poles_added +=1 # count the pole as added even if it is a predicted one

    if len(row_poles) >= row_size * 0.7: # at least 70% of the row size should be present
        return row_poles
    else:
        return None

all_rows = []
used_poles = np.zeros(len(pole_locations), dtype=bool)

for i, pole in enumerate(pole_locations):
    if not used_poles[i]:
      # Create a copy to avoid modifying the original list while iterating
        possible_row_sizes = num_poles_per_row[:]  
        for row_size in possible_row_sizes:
            potential_row = find_rows(pole, row_size)
            if potential_row is not None:
                all_rows.append(potential_row)
                for p in potential_row:
                    index = np.where((pole_locations == p).all(axis=1))[0][0]
                    used_poles[index] = True
                num_poles_per_row.remove(row_size) # Remove the size only AFTER successful row detection
                break

# --- Clustering to refine and remove outliers (Optional but recommended) ---
all_rows_np = [np.array(row) for row in all_rows if row is not None] # handle None values
if all_rows_np: # if there are rows detected
    all_points = np.concatenate(all_rows_np, axis=0)

    dbscan = DBSCAN(eps=2, min_samples=3)  # Adjust eps and min_samples as needed
    clusters = dbscan.fit_predict(all_points)

    filtered_rows = []
    for i, row in enumerate(all_rows_np):
        row_cluster_ids = clusters[np.where(np.isin(all_points, row).all(axis=1))[0]]
        if all(c != -1 for c in row_cluster_ids):
            filtered_rows.append(row)

    all_rows = filtered_rows

# --- Visualization (Optional) ---
if all_rows:
    for row in all_rows:
        row_x, row_y = zip(*row)
        plt.scatter(row_x, row_y, label='Row')  # Keep the scatter plot for poles

        # Connect the poles with lines
        for row in all_rows:
            row_x_proj, row_y_proj = zip(*row)  # Projected coordinates for plotting lines
            plt.plot(row_x_proj, row_y_proj, color='blue', linestyle='-', linewidth=1)

    plt.xlabel('Easting (m)')  # Update axis labels
    plt.ylabel('Northing (m)')
    plt.title('Detected Rows with Connections (Projected)')  # Update title

    # Plot original lat/lon points for reference (optional):
    # plt.scatter(gdf.geometry.x, gdf.geometry.y, c='red', s=10, label='Original Lat/Lon') # Uncomment if needed

    plt.legend()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()
    plt.savefig('../../images/geometric_pattern_matching/geometric_pattern_matching_with_lines_and_points_projected.png')
    plt.close()

# --- Output to GeoJSON ---
if all_rows:
    features = []
    for i, row in enumerate(all_rows):
        row_coordinates = [list(pole) for pole in row]  # Coordinates for LineString

        # Create a transformer to convert back to lon/lat
        transformer = Transformer.from_crs(target_crs, gdf.crs)

        # Linestring (in lon/lat) - Corrected order (lon, lat)
        row_lonlat = [transformer.transform(point[0], point[1]) for point in row_coordinates]
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": row_lonlat  # lon/lat coordinates
            },
            "properties": {
                "row_id": i,
                "type": "row_connection"
            }
        })

        # Points (in lon/lat) - Corrected order (lon, lat)
        for j, pole in enumerate(row):
            lonlat_point = transformer.transform(pole[0], pole[1])
            features.append({
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": lonlat_point
                },
                "properties": {
                    "row_id": i,
                    "pole_id": j
                }
            })

    geojson_output = {
        "type": "FeatureCollection",
        "features": features
    }

    with open("../../data/geometric_pattern_matching_detected_rows_projected.geojson", "w") as outfile:
        json.dump(geojson_output, outfile, indent=4)

    print("Detected rows (lines and points in lat/lon) saved to geometric_pattern_matching_detected_rows_projected.geojson")
else:
    print("No rows were detected.")