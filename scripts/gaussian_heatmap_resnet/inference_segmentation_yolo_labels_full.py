import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from PIL import Image
import numpy as np
import os
import json
import cv2
from tqdm import tqdm
from skimage.feature import peak_local_max
from skimage.morphology import skeletonize
from scipy.spatial import cKDTree
from shapely.geometry import Point as ShapelyPoint, LineString
from shapely.ops import nearest_points

# --- Import your GPS Utils ---
import image_gps_pixel_show_poles

# ======================
# CONFIGURATION
# ======================
MODEL_PATH = "results_resnet/yolo_masks/vineyard_segmentation_paper_1/train_resnet101_20260203_135036/resnet101_vineyard_segmentation_paper_1_unet_image_size_1280x960_batch_size_2.pth"
INPUT_DIR = "../../images/riseholme/august_2024/39_feet/"
OUTPUT_DIR = "resnet_inference/vineyard_segmentation_paper_1/full_images_2_filtered/train_resnet101_20260203_135036/inference_results_full/39_feet/"

BACKBONE = "resnet101"

# FULL RESOLUTION 
IMAGE_SIZE = (4056, 3040) 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Memory optimization
USE_MIXED_PRECISION = True  # Use FP16 for inference to save memory

# Thresholds
CONFIDENCE_THRESHOLDS = {
    "pole": 0.3,
    "trunk": 0.3,
    "vine_row": 0.3
}

# Deduplication Parameters (in meters)
ROW_SPACING_M = 2.5  # Distance between vine rows
POST_SPACING_M = 5.65  # Distance between posts along a row
POLE_DEDUP_RADIUS_M = POST_SPACING_M * 0.3  # 30% of post spacing
TRUNK_DEDUP_RADIUS_M = 0.5  # Trunks are closer together
ROW_ASSOCIATION_MAX_DIST_M = ROW_SPACING_M * 0.75  # Max distance to associate pole to row

# Camera Specs (Riseholme H20)
FOCAL_LENGTH_MM = 4.5
SENSOR_WIDTH_MM = 6.17
SENSOR_HEIGHT_MM = 4.55

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ======================
# MODEL DEFINITION
# ======================
class UNetResNet(nn.Module):
    def __init__(self, n_classes, backbone=BACKBONE):
        super().__init__()
        if backbone == "resnet18":
            resnet = models.resnet18(weights=None)
            enc_ch = [64, 64, 128, 256, 512] 
        elif backbone == "resnet50":
            resnet = models.resnet50(weights=None)
            enc_ch = [64, 256, 512, 1024, 2048]
        elif backbone == "resnet101":
            resnet = models.resnet101(weights=None)
            enc_ch = [64, 256, 512, 1024, 2048]
        
        self.encoder0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.pool0 = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.upconv4 = nn.ConvTranspose2d(enc_ch[4], enc_ch[3], 2, stride=2)
        self.dec4 = nn.Conv2d(enc_ch[3] + enc_ch[3], enc_ch[3], 3, padding=1)
        self.upconv3 = nn.ConvTranspose2d(enc_ch[3], enc_ch[2], 2, stride=2)
        self.dec3 = nn.Conv2d(enc_ch[2] + enc_ch[2], enc_ch[2], 3, padding=1)
        self.upconv2 = nn.ConvTranspose2d(enc_ch[2], enc_ch[1], 2, stride=2)
        self.dec2 = nn.Conv2d(enc_ch[1] + enc_ch[1], enc_ch[1], 3, padding=1)
        self.upconv1 = nn.ConvTranspose2d(enc_ch[1], enc_ch[0], 2, stride=2)
        self.dec1 = nn.Conv2d(enc_ch[0] + enc_ch[0], 64, 3, padding=1)
        self.final = nn.Conv2d(64, n_classes, 1)

    def forward(self, x):
        x0 = self.encoder0(x)
        x1 = self.pool0(x0)
        x2 = self.encoder1(x1)
        x3 = self.encoder2(x2)
        x4 = self.encoder3(x3)
        x5 = self.encoder4(x4)
        d4 = F.interpolate(self.upconv4(x5), size=x4.shape[2:], mode='bilinear', align_corners=True)
        d4 = torch.cat([d4, x4], dim=1)
        d4 = self.dec4(d4)
        d3 = F.interpolate(self.upconv3(d4), size=x3.shape[2:], mode='bilinear', align_corners=True)
        d3 = torch.cat([d3, x3], dim=1)
        d3 = self.dec3(d3)
        d2 = F.interpolate(self.upconv2(d3), size=x2.shape[2:], mode='bilinear', align_corners=True)
        d2 = torch.cat([d2, x2], dim=1)
        d2 = self.dec2(d2)
        d1 = F.interpolate(self.upconv1(d2), size=x0.shape[2:], mode='bilinear', align_corners=True)
        d1 = torch.cat([d1, x0], dim=1)
        d1 = self.dec1(d1)
        out = self.final(d1)
        out = F.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=True)
        return out

# ======================
# LOAD MODEL
# ======================
NUM_CLASSES = 4 
CLASS_NAMES = ["background", "pole", "trunk", "vine_row"]

model = UNetResNet(n_classes=NUM_CLASSES, backbone=BACKBONE).to(DEVICE)
try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print(f"✅ Model loaded: {MODEL_PATH}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit()

# ======================
# DEDUPLICATION UTILS
# ======================
def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate distance in meters between two GPS coordinates."""
    from math import radians, sin, cos, sqrt, atan2
    R = 6371000  # Earth radius in meters
    
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    return R * c

def deduplicate_points(points_list, radius_m):
    """Remove duplicate points that are within radius_m of each other.
    Keeps the point with highest confidence.
    points_list: [(lat, lon, confidence, other_props), ...]
    """
    if len(points_list) == 0:
        return []
    
    # Sort by confidence (descending)
    sorted_points = sorted(points_list, key=lambda x: x[2], reverse=True)
    kept = []
    
    for point in sorted_points:
        lat, lon, conf = point[0], point[1], point[2]
        
        # Check if too close to any kept point
        is_duplicate = False
        for kept_point in kept:
            dist = haversine_distance(lat, lon, kept_point[0], kept_point[1])
            if dist < radius_m:
                is_duplicate = True
                break
        
        if not is_duplicate:
            kept.append(point)
    
    return kept

def associate_poles_to_rows(poles_gps, rows_gps, max_dist_m):
    """Associate each pole to its nearest vine row.
    Returns: dict {pole_idx: row_idx} and dict {row_idx: [pole_indices]}
    """
    from shapely.geometry import Point as ShapelyPoint, LineString
    from shapely.ops import nearest_points
    
    pole_to_row = {}
    row_to_poles = {i: [] for i in range(len(rows_gps))}
    
    for p_idx, (p_lat, p_lon, p_conf, p_props) in enumerate(poles_gps):
        pole_point = ShapelyPoint(p_lon, p_lat)
        
        min_dist = float('inf')
        closest_row = None
        
        for r_idx, row_coords in enumerate(rows_gps):
            if len(row_coords) < 2:
                continue
            row_line = LineString([(lon, lat) for lat, lon in row_coords])
            dist = pole_point.distance(row_line)
            
            # Convert degrees to approximate meters (rough estimate at mid-latitudes)
            dist_m = dist * 111320 * np.cos(np.radians(p_lat))
            
            if dist_m < min_dist:
                min_dist = dist_m
                closest_row = r_idx
        
        # Only associate if within max distance
        if closest_row is not None and min_dist < max_dist_m:
            pole_to_row[p_idx] = closest_row
            row_to_poles[closest_row].append(p_idx)
    
    return pole_to_row, row_to_poles

# ======================
# VISUALIZATION UTILS
# ======================
def save_heatmap_overlay(image_np, prob_map, peaks_list, output_path, color_map=cv2.COLORMAP_JET, alpha=0.5):
    """
    Creates a heatmap overlay AND draws the detected peak points on top.
    """
    # 1. Create Heatmap
    # Normalize probabilities to 0-255
    heatmap_uint8 = (prob_map * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, color_map)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    
    # 2. Blend with Original Image
    # Create mask where probability is significant (> 10%) to keep background clear
    mask = prob_map > 0.1
    overlay = image_np.copy()
    
    # Blend only where mask is true
    if np.any(mask):
        blended = cv2.addWeighted(image_np, 1 - alpha, heatmap_color, alpha, 0)
        overlay[mask] = blended[mask]

    # 3. Draw "X" marks on the detected peaks
    # This helps verify that the GPS logic matches the visual blob
    for x, y in peaks_list:
        # Draw a white X with a black outline for visibility
        cv2.drawMarker(overlay, (x, y), (0, 0, 0), markerType=cv2.MARKER_CROSS, markerSize=15, thickness=3)
        cv2.drawMarker(overlay, (x, y), (255, 255, 255), markerType=cv2.MARKER_CROSS, markerSize=15, thickness=1)

    # 4. Save
    Image.fromarray(overlay).save(output_path)

# ======================
# MAIN LOOP
# ======================
image_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.jpg', '.png'))]

# Store all detections before deduplication
all_poles = []  # [(lat, lon, confidence, props), ...]
all_trunks = []  # [(lat, lon, confidence, props), ...]
all_rows = []  # [[(lat, lon), (lat, lon), ...], ...]  # List of line coordinates per image

print(f"Processing {len(image_files)} images...")
print(f"Using device: {DEVICE}, Mixed Precision: {USE_MIXED_PRECISION}")

for file in tqdm(image_files):
    img_path = os.path.join(INPUT_DIR, file)
    
    # --- 1. Load EXIF ---
    (flight_yaw, flight_pitch, flight_roll, gimbal_yaw, gimbal_pitch, gimbal_roll, 
     gps_lat, gps_lon, gps_alt, fov, _, img_h, img_w) = image_gps_pixel_show_poles.extract_exif(img_path)

    if gps_lat is None:
        print(f"Skipping {file}: No GPS data found.")
        continue

    flight_yaw_num = image_gps_pixel_show_poles.extract_number(flight_yaw)
    gimbal_yaw_num = image_gps_pixel_show_poles.extract_number(gimbal_yaw)
    if gimbal_yaw_num == 0.0 or gimbal_yaw_num is None: gimbal_yaw_num = flight_yaw_num
    
    gps_alt_num = image_gps_pixel_show_poles.extract_number(gps_alt)
    if gps_alt_num is None: gps_alt_num = 0.0

    # --- 2. Inference ---
    image_pil = Image.open(img_path).convert("RGB")
    if image_pil.size != IMAGE_SIZE:
        image_pil = image_pil.resize(IMAGE_SIZE, Image.BILINEAR)
        
    image_np = np.array(image_pil)
    image_tensor_in = image_np / 255.0
    image_tensor = torch.from_numpy(image_tensor_in).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE)

    with torch.no_grad():
        if USE_MIXED_PRECISION and DEVICE.type == 'cuda':
            with torch.cuda.amp.autocast():
                outputs = model(image_tensor)
        else:
            outputs = model(image_tensor)
        
        probs = torch.softmax(outputs, dim=1)[0].cpu().numpy()
        
        # Clear GPU cache to free memory
        del outputs, image_tensor
        if DEVICE.type == 'cuda':
            torch.cuda.empty_cache()

    # --- 3. Feature Extraction & Visualization ---
    
    # Define Indices
    pole_idx = CLASS_NAMES.index("pole")
    trunk_idx = CLASS_NAMES.index("trunk")
    row_idx = CLASS_NAMES.index("vine_row")

    # A. Poles
    pole_coords = peak_local_max(probs[pole_idx], min_distance=20, threshold_abs=CONFIDENCE_THRESHOLDS["pole"])
    pole_pixels = pole_coords[:, ::-1] # (y, x) -> (x, y)
    
    save_heatmap_overlay(
        image_np, probs[pole_idx], pole_pixels, 
        os.path.join(OUTPUT_DIR, f"{file}_pole_vis.jpg"), 
        color_map=cv2.COLORMAP_AUTUMN # Red/Orange for Poles
    )

    # B. Trunks
    trunk_coords = peak_local_max(probs[trunk_idx], min_distance=20, threshold_abs=CONFIDENCE_THRESHOLDS["trunk"])
    trunk_pixels = trunk_coords[:, ::-1]

    save_heatmap_overlay(
        image_np, probs[trunk_idx], trunk_pixels, 
        os.path.join(OUTPUT_DIR, f"{file}_trunk_vis.jpg"),
        color_map=cv2.COLORMAP_WINTER # Blue/Green for Trunks
    )

    # C. Vine Rows (Skeletonization)
    row_prob_map = probs[row_idx]
    row_binary = row_prob_map > CONFIDENCE_THRESHOLDS["vine_row"]
    row_skeleton = skeletonize(row_binary)
    row_y, row_x = np.where(row_skeleton)
    row_pixels = np.column_stack((row_x, row_y))

    # Visualization for Rows (Overlay the green skeleton)
    row_vis = image_np.copy()
    # Make the skeleton bright green and thick
    # Dilate slightly for visibility
    skel_uint8 = (row_skeleton * 255).astype(np.uint8)
    skel_dilated = cv2.dilate(skel_uint8, np.ones((3,3), np.uint8))
    row_vis[skel_dilated > 0] = [0, 255, 0] 
    Image.fromarray(row_vis).save(os.path.join(OUTPUT_DIR, f"{file}_row_vis.jpg"))

    # --- 4. GPS Conversion ---
    def to_gps(px, py):
        return image_gps_pixel_show_poles.get_gps_from_pixel(
            px, py, IMAGE_SIZE[0], IMAGE_SIZE[1], 
            flight_yaw_num, gimbal_yaw_num, 
            gps_lat, gps_lon, gps_alt_num, 
            FOCAL_LENGTH_MM, SENSOR_WIDTH_MM, SENSOR_HEIGHT_MM
        )

    # Collect Poles
    for x, y in pole_pixels:
        lat, lon = to_gps(x, y)
        confidence = float(probs[pole_idx, y, x])
        all_poles.append((lat, lon, confidence, {"image": file}))

    # Collect Trunks
    for x, y in trunk_pixels:
        lat, lon = to_gps(x, y)
        confidence = float(probs[trunk_idx, y, x])
        all_trunks.append((lat, lon, confidence, {"image": file}))

    # Collect Rows
    if len(row_pixels) > 0:
        row_gps_points = []
        for x, y in row_pixels[::15]: 
            lat, lon = to_gps(x, y)
            row_gps_points.append((lat, lon))
        all_rows.append(row_gps_points)

print("\n🔄 Deduplicating detections...")

# Deduplicate poles and trunks
unique_poles = deduplicate_points(all_poles, POLE_DEDUP_RADIUS_M)
unique_trunks = deduplicate_points(all_trunks, TRUNK_DEDUP_RADIUS_M)

print(f"  Poles: {len(all_poles)} → {len(unique_poles)} (removed {len(all_poles) - len(unique_poles)} duplicates)")
print(f"  Trunks: {len(all_trunks)} → {len(unique_trunks)} (removed {len(all_trunks) - len(unique_trunks)} duplicates)")
print(f"  Vine Rows: {len(all_rows)} detected")

print("\n🔗 Associating poles to vine rows...")

# Associate poles to rows
pole_to_row, row_to_poles = associate_poles_to_rows(unique_poles, all_rows, ROW_ASSOCIATION_MAX_DIST_M)

print(f"  Associated {len(pole_to_row)}/{len(unique_poles)} poles to rows")

# ======================
# CREATE SEPARATE GEOJSON FILES
# ======================
print("\n💾 Creating GeoJSON outputs...")

# 1. POLES GeoJSON
geojson_poles = {"type": "FeatureCollection", "features": []}
for p_idx, (lat, lon, conf, props) in enumerate(unique_poles):
    row_id = pole_to_row.get(p_idx, None)
    geojson_poles["features"].append({
        "type": "Feature",
        "geometry": {"type": "Point", "coordinates": [lon, lat]},
        "properties": {
            "confidence": conf,
            "row_id": row_id,
            "image": props["image"]
        }
    })

# 2. TRUNKS GeoJSON
geojson_trunks = {"type": "FeatureCollection", "features": []}
for lat, lon, conf, props in unique_trunks:
    geojson_trunks["features"].append({
        "type": "Feature",
        "geometry": {"type": "Point", "coordinates": [lon, lat]},
        "properties": {
            "confidence": conf,
            "image": props["image"]
        }
    })

# 3. VINE ROWS GeoJSON
geojson_rows = {"type": "FeatureCollection", "features": []}
for r_idx, row_coords in enumerate(all_rows):
    if len(row_coords) >= 2:
        geojson_rows["features"].append({
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": [[lon, lat] for lat, lon in row_coords]
            },
            "properties": {
                "row_id": r_idx,
                "num_poles": len(row_to_poles.get(r_idx, []))
            }
        })

# 4. POLE CONNECTIONS (Poles in same row connected by lines)
geojson_pole_connections = {"type": "FeatureCollection", "features": []}
for row_id, pole_indices in row_to_poles.items():
    if len(pole_indices) >= 2:
        # Sort poles by position along the row
        poles_in_row = [unique_poles[p_idx] for p_idx in pole_indices]
        # Simple sorting by latitude (adjust based on row orientation)
        poles_in_row_sorted = sorted(poles_in_row, key=lambda p: (p[0], p[1]))
        
        # Create line connecting all poles in this row
        pole_line_coords = [[p[1], p[0]] for p in poles_in_row_sorted]  # [lon, lat]
        geojson_pole_connections["features"].append({
            "type": "Feature",
            "geometry": {"type": "LineString", "coordinates": pole_line_coords},
            "properties": {
                "row_id": row_id,
                "num_poles": len(pole_indices)
            }
        })

# Save all GeoJSON files
with open(os.path.join(OUTPUT_DIR, "poles.geojson"), "w") as f:
    json.dump(geojson_poles, f, indent=2)

with open(os.path.join(OUTPUT_DIR, "trunks.geojson"), "w") as f:
    json.dump(geojson_trunks, f, indent=2)

with open(os.path.join(OUTPUT_DIR, "vine_rows.geojson"), "w") as f:
    json.dump(geojson_rows, f, indent=2)

with open(os.path.join(OUTPUT_DIR, "pole_connections.geojson"), "w") as f:
    json.dump(geojson_pole_connections, f, indent=2)

print(f"\n✅ Finished! Results saved to {OUTPUT_DIR}")
print(f"  - poles.geojson ({len(unique_poles)} poles)")
print(f"  - trunks.geojson ({len(unique_trunks)} trunks)")
print(f"  - vine_rows.geojson ({len(all_rows)} rows)")
print(f"  - pole_connections.geojson ({len(geojson_pole_connections['features'])} row connections)")