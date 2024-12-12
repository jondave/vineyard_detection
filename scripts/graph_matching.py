import geojson
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch  # for plotting matching result

import math

import numpy as np  # numpy backend
import pygmtools as pygm

def haversine_distance(coord1, coord2):
    """
    Calculates the Haversine distance between two geographic coordinates.

    Args:
        coord1: A tuple of (latitude1, longitude1) in decimal degrees.
        coord2: A tuple of (latitude2, longitude2) in decimal degrees.

    Returns:
        The distance between the two points in kilometers.
    """

    R = 6371  # Earth radius in kilometers

    lat1, lon1 = coord1
    lat2, lon2 = coord2

    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

    distance = R * c
    return distance

# Suggested distance threshold (adjust as needed based on your specific use case)
distance_threshold = 50  # meters

def create_detection_graph(geojson_file):
    """Creates a graph from detected pole locations in a GeoJSON file."""
    with open(geojson_file) as f:
        data = geojson.load(f)

    G = nx.Graph()
    for feature in data['features']:
        coords = tuple(feature['geometry']['coordinates'])
        G.add_node(coords, pos=coords)  # Assign coordinates as node attribute

    # # Add edges based on proximity (adjust distance threshold as needed)
    # for u, data1 in G.nodes(data=True):
    #     for v, data2 in G.nodes(data=True):
    #         if u != v and haversine_distance(u, v) < distance_threshold:
    #             G.add_edge(u, v)

    return G

def create_grid_graph(num_rows, num_cols, row_spacing, col_spacing):
    """
    Creates a grid graph where nodes are placed in a regular grid layout
    with specified spacing between rows and columns.

    Args:
        num_rows (int): Number of rows in the grid.
        num_cols (int): Number of columns in the grid.
        row_spacing (float): Spacing between rows (y-axis).
        col_spacing (float): Spacing between columns (x-axis).

    Returns:
        G (networkx.Graph): A grid graph.
    """
    G = nx.Graph()

    for row in range(num_rows):
        for col in range(num_cols):
            x = col * col_spacing
            y = row * row_spacing
            node = (x, y)
            G.add_node(node, pos=(x, y))

            # Connect nodes within the grid
            if row > 0:  # Connect to the node above
                G.add_edge((col * col_spacing, (row - 1) * row_spacing), node)
            if col > 0:  # Connect to the node to the left
                G.add_edge(((col - 1) * col_spacing, row * row_spacing), node)

    return G


# Suggested distance threshold (adjust as needed based on your specific use case)
distance_threshold = 1  # meters

def visualize_graph(G):
    """Visualizes the graph, preserving spatial relationships."""
    pos = nx.spring_layout(G, pos=nx.get_node_attributes(G, 'pos'))
    nx.draw_networkx(G, pos=pos, node_color='blue', node_size=100, with_labels=False)
    plt.axis('off')
    plt.savefig('../images/pole_graph.png')

    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    print("Number of nodes:", num_nodes)
    print("Number of edges:", num_edges)

# Example usage:
geojson_file = '../data/clustered_poles.geojson'

# Create the graph
G = create_detection_graph(geojson_file)

# Visualize and save the graph
# visualize_graph(G)

# Create the prior knowledge graph
num_rows = 10
num_cols = 4 # number of poles per row
row_spacing = 2.75
col_spacing = 5.65 # pole spacing

G_prior = create_grid_graph(num_rows, num_cols, row_spacing, col_spacing)

K = pygm.utils.build_aff_mat_from_networkx(G, G_prior)

X = pygm.rrwm(K, G.number_of_nodes(), G_prior.number_of_nodes())

X = pygm.hungarian(X)

# Visualize both graphs
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
pos1 = nx.get_node_attributes(G, 'pos')
nx.draw_networkx(G, pos=pos1, node_color='blue', node_size=100, with_labels=False)
plt.title("Detected Pole Graph")

plt.subplot(1, 2, 2)
pos2 = nx.get_node_attributes(G_prior, 'pos')
nx.draw_networkx(G_prior, pos=pos2, node_color='green', node_size=100, with_labels=False)
plt.title("Prior Knowledge Graph")
plt.savefig('../images/pole_graph.png')