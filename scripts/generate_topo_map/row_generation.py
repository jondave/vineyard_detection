import math
from typing import Dict, List, Tuple

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans


def _extract_vine_row_direction(
    vine_rows_geojson: Dict,
) -> Tuple[np.ndarray, bool]:
    """
    Estimate row direction from detected vine row polygons.
    Vine row polygons are PERPENDICULAR to the actual row direction.
    We use the MINOR axis (smallest eigenvalue) as the row direction.
    Returns (direction_vector, success_flag).
    """
    features = vine_rows_geojson.get("features", [])
    if not features:
        return np.array([1.0, 0.0]), False

    all_points = []
    for feat in features:
        if feat.get("geometry", {}).get("type") != "Polygon":
            continue
        coords = feat.get("geometry", {}).get("coordinates", [[]])
        all_points.extend(coords[0])

    if len(all_points) < 3:
        return np.array([1.0, 0.0]), False

    lons = np.array([c[0] for c in all_points], dtype=np.float64)
    lats = np.array([c[1] for c in all_points], dtype=np.float64)
    center_lon = float(np.mean(lons))
    center_lat = float(np.mean(lats))

    lat_to_m = 111111.0
    lon_to_m = 111111.0 * np.cos(np.radians(center_lat))
    xy = np.column_stack([
        (lons - center_lon) * lon_to_m,
        (lats - center_lat) * lat_to_m,
    ])

    cov = np.cov(xy.T)
    try:
        eigvals, eigvecs = np.linalg.eig(cov)
        # Vine rows are PERPENDICULAR to the actual direction,
        # so use the MINOR axis (smallest eigenvalue)
        minor_idx = int(np.argmin(eigvals))
        minor = eigvecs[:, minor_idx]
        norm = np.linalg.norm(minor)
        if norm > 1e-6:
            direction = minor / norm
            return direction, True
    except Exception:
        pass

    return np.array([1.0, 0.0]), False


def _meters_from_lonlat(lon: float, lat: float, center_lon: float, center_lat: float) -> Tuple[float, float]:
    lat_to_m = 111111.0
    lon_to_m = 111111.0 * math.cos(math.radians(center_lat))
    x_dist = (lon - center_lon) * lon_to_m
    y_dist = (lat - center_lat) * lat_to_m
    return x_dist, y_dist


def _dominant_direction(points_xy: np.ndarray) -> np.ndarray:
    if points_xy.shape[0] < 2:
        return np.array([1.0, 0.0])
    cov = np.cov(points_xy.T)
    eigvals, eigvecs = np.linalg.eig(cov)
    principal = eigvecs[:, int(np.argmax(eigvals))]
    norm = np.linalg.norm(principal)
    if norm == 0:
        return np.array([1.0, 0.0])
    return principal / norm


def _estimate_row_spacing(perp_coords: np.ndarray) -> float:
    if perp_coords.size < 2:
        return 2.0
    unique_vals = np.unique(np.round(perp_coords, 3))
    if unique_vals.size < 2:
        return 2.0
    diffs = np.diff(np.sort(unique_vals))
    diffs = diffs[diffs > 0.2]
    if diffs.size == 0:
        return 2.0
    return float(np.median(diffs))


def _cluster_row_labels_dbscan(perp_coords: np.ndarray) -> np.ndarray:
    row_spacing = _estimate_row_spacing(perp_coords)
    eps = max(row_spacing * 0.75, 1.0)
    clustering = DBSCAN(eps=eps, min_samples=2).fit(perp_coords.reshape(-1, 1))
    return clustering.labels_


def generate_rows(
    poles_geojson: Dict,
    vine_rows_geojson: Dict | None = None,
    row_direction_mode: str = "auto",
) -> Dict:
    """
    Generate vine rows (LineStrings) from clustered poles.
    Direction modes:
    - auto: prefer vine_rows_geojson direction if available, else PCA from poles
    - north_south: force a general north-south row direction
    - perpendicular_auto: force 90 degrees to the auto direction
    """
    features = poles_geojson.get("features", [])
    if len(features) < 2:
        return {"type": "FeatureCollection", "features": []}

    coords = np.array([f["geometry"]["coordinates"] for f in features], dtype=np.float64)
    lons = coords[:, 0]
    lats = coords[:, 1]
    center_lon = float(np.mean(lons))
    center_lat = float(np.mean(lats))

    xy = np.array([
        _meters_from_lonlat(lon, lat, center_lon, center_lat)
        for lon, lat in coords
    ], dtype=np.float64)

    direction_mode = (row_direction_mode or "auto").lower()
    direction = None
    direction_source = "pole_pca"
    auto_direction = None
    auto_row_count = None

    if direction_mode == "north_south":
        # Local-meter frame: +Y is geographic north.
        direction = np.array([0.0, 1.0], dtype=np.float64)
        direction_source = "manual_north_south"
    else:
        used_vine_rows = False
        if vine_rows_geojson:
            auto_direction, used_vine_rows = _extract_vine_row_direction(vine_rows_geojson)

        if used_vine_rows:
            direction_source = "vine_rows"
        else:
            auto_direction = _dominant_direction(xy)
            direction_source = "pole_pca"

        if direction_mode == "perpendicular_auto":
            direction = np.array([-auto_direction[1], auto_direction[0]], dtype=np.float64)
            if used_vine_rows:
                direction_source = "perpendicular_vine_rows"
            else:
                direction_source = "perpendicular_pole_pca"

            # Estimate how many perpendicular rows we should have by first
            # counting the auto-oriented rows, then transposing the grid.
            auto_perp = np.array([-auto_direction[1], auto_direction[0]])
            auto_perp_coords = xy @ auto_perp
            auto_labels = _cluster_row_labels_dbscan(auto_perp_coords)
            auto_row_count = len({int(label) for label in auto_labels if int(label) != -1})
        else:
            direction = auto_direction

    perp = np.array([-direction[1], direction[0]])
    perp_coords = xy @ perp
    proj_coords = xy @ direction

    if direction_mode == "perpendicular_auto" and auto_row_count and auto_row_count > 0:
        expected_rows = max(2, int(round(len(features) / auto_row_count)))
        expected_rows = min(expected_rows, len(features))
        labels = KMeans(n_clusters=expected_rows, n_init=20, random_state=42).fit_predict(perp_coords.reshape(-1, 1))
    else:
        labels = _cluster_row_labels_dbscan(perp_coords)

    line_features: List[Dict] = []
    for label in sorted(set(labels)):
        if label == -1:
            continue
        indices = np.where(labels == label)[0]
        if indices.size < 2:
            continue

        # Connect the actual pole positions in order along the chosen direction.
        order = indices[np.argsort(proj_coords[indices])]
        line_coords = coords[order].tolist()

        line_features.append({
            "type": "Feature",
            "geometry": {"type": "LineString", "coordinates": line_coords},
            "properties": {"row_id": int(label), "direction_source": direction_source},
        })

    return {"type": "FeatureCollection", "features": line_features}
