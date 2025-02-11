import networkx as nx
import numpy as np
from scipy.spatial import KDTree
from scipy.optimize import linear_sum_assignment
from sklearn.manifold import MDS
import matplotlib.pyplot as plt
import geojson
from scipy.spatial import Delaunay
from sklearn.decomposition import PCA
from itertools import permutations
import geopandas as gpd
from pyproj import CRS

def create_delaunay_graph(G: nx.Graph) -> nx.Graph:
    """
    Create a Delaunay triangulation graph from an input graph G.
    
    Parameters:
    G (nx.Graph): Input graph where nodes have 'pos' attributes (2D coordinates).
    
    Returns:
    nx.Graph: A new graph with the same nodes but edges determined by Delaunay triangulation.
    """
    # Extract node positions
    node_positions = {node: data['pos'] for node, data in G.nodes(data=True)}
    points = np.array(list(node_positions.values()))
    
    # Compute Delaunay triangulation
    tri = Delaunay(points)

    # Create a new graph with Delaunay edges
    G_delaunay = nx.Graph()
    G_delaunay.add_nodes_from(G.nodes(data=True))

    # Add edges based on Delaunay triangulation
    node_list = list(node_positions.keys())  # Ensure we reference correct node IDs
    for simplex in tri.simplices:
        for i in range(3):  # Each triangle has 3 nodes
            n1, n2 = node_list[simplex[i]], node_list[simplex[(i + 1) % 3]]
            G_delaunay.add_edge(n1, n2)

    return G_delaunay

def create_prior_knowledge_graph_with_variable_poles(num_poles_per_col, row_spacing, col_spacing):
    """
    Creates a graph where each column can have a different number of poles (nodes).

    Args:
        num_poles_per_col (list or int): A list specifying the number of poles in each column,
                                         or a single integer for a uniform number of poles per column.
        row_spacing (float): Spacing between rows (y-axis).
        col_spacing (float): Spacing between columns (x-axis).

    Returns:
        G (networkx.Graph): A graph with nodes placed in a layout with varying poles per column.
    """
    G = nx.Graph()

    # Ensure num_poles_per_col is a list
    if isinstance(num_poles_per_col, int):
        num_poles_per_col = [num_poles_per_col] * int(col_spacing / row_spacing)

    for col, num_rows in enumerate(num_poles_per_col):
        for row in range(num_rows):
            x = col * col_spacing
            y = row * row_spacing
            name = f"Node_{row}_{col}"  # Assign a name based on the row and column
            node = (x, y)
            G.add_node(node, pos=(x, y), name=name)

            # Connect nodes within the same column
            if row > 0:  # Connect to the node above (same column)
                G.add_edge(((col) * col_spacing, (row - 1) * row_spacing), node)

            # # Connect nodes within the same row (if a node exists to the left in this row)
            # if col > 0 and row < num_poles_per_col[col - 1]:  # Ensure row exists in the previous column
            #     G.add_edge(((col - 1) * col_spacing, row * row_spacing), node)

    return G

def create_detection_graph_cartesian(geojson_file):
    """
    Converts GeoJSON coordinates (longitude, latitude) in a graph to Cartesian coordinates (x, y),
    where the southwest-most node becomes the origin (0, 0).

    Args:
        geojson_file (str): Path to the GeoJSON file with the pole locations.

    Returns:
        G (networkx.Graph): The resulting graph with Cartesian coordinates.
    """
    with open(geojson_file) as f:
        data = geojson.load(f)

    G = nx.Graph()

    # Add nodes and find the southwest-most coordinate
    coords_list = []
    for feature in data['features']:
        coords = tuple(feature['geometry']['coordinates'])
        coords_list.append(coords)
        G.add_node(coords, pos=coords)  # Assign original coordinates as a node attribute

    # Southwest-most node as the origin
    min_lon, min_lat = min(coords_list, key=lambda c: (c[1], c[0]))

    # Conversion to Cartesian coordinates
    def latlon_to_xy(lon, lat, origin_lon, origin_lat):
        """Converts (lon, lat) to (x, y) relative to an origin point."""
        from pyproj import Proj, transform

        # Define projection: WGS84 to a local projected system
        proj_wgs84 = Proj(proj='latlong', datum='WGS84')
        proj_local = Proj(proj='aeqd', lat_0=origin_lat, lon_0=origin_lon)  # Azimuthal Equidistant

        x, y = transform(proj_wgs84, proj_local, lon, lat)
        return x, y

    # Replace coordinates in the graph
    for node in list(G.nodes()):
        lon, lat = node
        x, y = latlon_to_xy(lon, lat, min_lon, min_lat)
        nx.set_node_attributes(G, {node: (x, y)}, 'pos')
        G = nx.relabel_nodes(G, {node: (x, y)})

    # # Add edges based on proximity (adjust distance threshold as needed)
    # for u in G.nodes():
    #     for v in G.nodes():
    #         if u != v:
    #             # Calculate Euclidean distance between the two nodes
    #             distance = math.sqrt((u[0] - v[0])**2 + (u[1] - v[1])**2)
    #             if distance < distance_threshold_max and distance > distance_threshold_min:
    #                 G.add_edge(u, v)

    return G

def verify_geometry(matched_nodes, pole_locations, ideal_spacing, tolerance=0.5):
    """Verifies if matched nodes form a linear arrangement with correct spacing."""

    if len(matched_nodes) < 2:  # Need at least 2 points to check for a line
        return False

    # 1. Retrieve Node Positions
    node_positions = [pole_locations[node] for node in matched_nodes]
    node_positions = np.array(node_positions)

    # 2. Check for Linear Arrangement (using PCA)
    pca = PCA(n_components=2)
    pca.fit(node_positions)
    principal_direction = pca.components_[0]  # Get the direction of the longest principal component

    # Calculate perpendicular distances (simplified for 2D)
    # Assuming principal_direction is normalized
    distances = np.abs(np.dot(node_positions - pca.mean_, principal_direction))

    if np.any(distances > tolerance):  # Check if any point is too far
        return False

    # 3. Check Spacing (crude check for demonstration)
    distances_between_nodes = []
    for i in range(len(node_positions) - 1):
        distance = np.linalg.norm(node_positions[i+1] - node_positions[i])
        distances_between_nodes.append(distance)

    average_distance = np.mean(distances_between_nodes) if distances_between_nodes else 0

    if abs(average_distance - ideal_spacing) > tolerance:
      return False

    return True  # Both checks passed

def match_graphs(G1, G2, pole_locations):
    """Approximate graph matching with geometric verification."""

    best_match = None
    min_edit_distance = float('inf')  # For graph edit distance

    for subgraph_nodes_G2 in nx.find_cliques(G2):  # Iterate through maximal cliques (potential rows)
        subgraph = G2.subgraph(subgraph_nodes_G2)

        # Only consider subgraphs with a reasonable size (at least 70% of template size)
        if subgraph.number_of_nodes() >= G1.number_of_nodes() * 0.7:
            for perm in permutations(subgraph.nodes()):  # Iterate through all permutations
                subgraph_perm = subgraph.copy()
                nodes = list(subgraph_perm.nodes())
                for i in range(len(nodes)):
                    subgraph_perm.remove_node(nodes[i])
                    subgraph_perm.add_node(perm[i])

                edit_distance = nx.graph_edit_distance(subgraph_perm, G1)  # Graph edit distance

                if edit_distance < min_edit_distance:
                    min_edit_distance = edit_distance
                    best_match = subgraph_perm, perm  # Store the best match and permutation

            if best_match is not None and min_edit_distance <= 2:  # Tolerance on the edit distance
                best_subgraph, best_perm = best_match
                row_nodes = list(best_subgraph.nodes())

                # Geometric Verification (Crucial!)
                if verify_geometry(row_nodes, pole_locations, pole_spacing):  # Pass pole_locations
                    row_nodes_g1 = []
                    for n2 in row_nodes:
                        for n1 in G1.nodes():
                            if G1.nodes[n1]['name'] == G2.nodes[n2]['name']:  # Compare node names
                                row_nodes_g1.append(n1)
                                break

                    return dict(zip(row_nodes_g1, row_nodes))  # Return a dict mapping from G1 to G2

    return None  # No suitable match found

def apply_transformation(G1, matched_G1, matched_G2):
    """Compute transformation (rotation + translation) to align G1 to G2"""
    source_points = np.array([G1.nodes[n]['pos'] for n in matched_G1])
    target_points = np.array([G2.nodes[n]['pos'] for n in matched_G2])

    # Compute optimal rotation and translation
    centroid_src = np.mean(source_points, axis=0)
    centroid_tgt = np.mean(target_points, axis=0)
    
    centered_src = source_points - centroid_src
    centered_tgt = target_points - centroid_tgt
    
    U, _, Vt = np.linalg.svd(centered_tgt.T @ centered_src)
    R = U @ Vt
    t = centroid_tgt - R @ centroid_src

    # Apply transformation
    transformed_G1 = G1.copy()
    for node in transformed_G1.nodes:
        transformed_G1.nodes[node]['pos'] = (R @ G1.nodes[node]['pos']) + t

    return transformed_G1

def plot_graph(G, file_name, title="Graph"):
    """Visualise graph with positions"""
    pos = nx.get_node_attributes(G, 'pos')
    nx.draw(G, pos, with_labels=False, node_color='lightblue', edge_color='gray')
    plt.title(title)
    plt.show()

    plt.savefig(file_name)
    plt.close()

# Create Graphs
# G1 = nx.cycle_graph(5)  # Prior knowledge graph (known edges, unknown positions)
# G2 = nx.Graph()  # Detected graph (known positions, no edges)
# for i in range(5):
#     G2.add_node(i, pos=(np.random.rand(), np.random.rand()))

# Riseholme
# row_spacing = 5.65 # pole spacing meters along the row
# col_spacing = 2.75 # meters between rows
# num_poles_per_col = [4, 4, 4, 4, 4, 4, 4, 4, 4, 4] # Riseholme
# geojson_file = '../data/clustered_poles.geojson'

# JoJo
row_spacing = 5.00 # pole spacing meters along the row
col_spacing = 3.00 # meters between rows
num_poles_per_col = [5, 7, 8, 9, 10, 11, 13, 14, 15, 16]
geojson_file = '../data/jojo_row_posts_10_rows.geojson'

G1 = create_prior_knowledge_graph_with_variable_poles(num_poles_per_col, row_spacing, col_spacing) # in long, lat coordinates with variable poles per row
plot_graph(G1, "../images/graph_matching/1_prior_g1", "Prior Knowledge G1")

G2 = create_detection_graph_cartesian(geojson_file) # in x, y coordinates
G2 = create_delaunay_graph(G2)

positions1 = nx.get_node_attributes(G1, 'pos')
nx.set_node_attributes(G1, positions1, 'pos')

positions2 = nx.get_node_attributes(G2, 'pos')

# gdf = gpd.read_file(geojson_file)

# # Define the target CRS (UTM zone) - VERY IMPORTANT!
# target_crs = CRS.from_string("EPSG:4326")  # Replace with the correct EPSG code for your UTM zone
# gdf_projected = gdf.to_crs(target_crs)  # Project to the correct CRS

# # 2. Create pole_locations (from PROJECTED coordinates)
# pole_locations = np.array(list(zip(gdf_projected.geometry.x, gdf_projected.geometry.y)))

# Match graphs
matched_nodes = match_graphs(G1, G2, positions2)

# Check if a match was found before proceeding
if matched_nodes is not None:  # Handle the case where no match is found
    # Apply transformation
    aligned_G1 = apply_transformation(G1, matched_nodes.keys(), matched_nodes.values())

    # Plot results
    plot_graph(G1, "../images/graph_matching/1_prior_g1", "Prior Knowledge G1")
    plot_graph(G2, "../images/graph_matching/2_detected_g2", "Detected G2")
    plot_graph(aligned_G1, "../images/graph_matching/3_aligned_g1", "Aligned G1")
else:
    print("No suitable match found between the graphs.")
