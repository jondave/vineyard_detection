"""
Example script to cluster pole detections with typical vineyard parameters.
"""

from cluster_poles import cluster_poles

# Path to your GeoJSON file
input_file = "data/riseholme/vineyard_segmentation-22/train5_inference_results/august_2024/39_feet/detected_pole_coordinates.geojson"

# Typical vineyard parameters (adjust based on your vineyard)
POLE_SPACING = 5.65  # meters between poles in a row
ROW_SPACING = 2.5   # meters between rows (not used in clustering, but good to know)

# Run clustering
# For overlapping images, we want to merge only very close duplicates
# Use row_spacing as reference since that's the minimum distance between different poles
# Let's use 1.0m radius (well under the 2.5m row spacing)
results = cluster_poles(
    geojson_path=input_file,
    pole_spacing=1.0,  # Use a small fixed distance instead
    cluster_eps_factor=1.0,  # Makes eps = 1.0m
    min_samples=1,  # Any point can start a cluster
    verbose=True
)

print("\n" + "="*60)
print("CLUSTERING RESULTS")
print("="*60)
print(f"Reduction: {results['original_count']} → {results['clustered_count']} poles")
print(f"Removed: {results['original_count'] - results['clustered_count']} duplicate detections")
print(f"\n✓ Clustered poles saved as GeoJSON to:")
print(f"  {results['output_path']}")
