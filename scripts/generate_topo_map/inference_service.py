import os
import sys
import json
import uuid
import datetime
import math
import time
from typing import Dict, List, Tuple, Optional, Union
from abc import ABC, abstractmethod

import cv2
import numpy as np
from PIL import Image
from skimage.feature import peak_local_max
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# Try importing YOLO
try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SCRIPTS_DIR = os.path.join(BASE_DIR, "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.append(SCRIPTS_DIR)

from image_gps_pixel_show_poles import extract_exif, extract_number, get_gps_from_pixel

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff")

FOCAL_LENGTH_MM = 4.5
SENSOR_WIDTH_MM = 6.17
SENSOR_HEIGHT_MM = 4.55

DEFAULT_IMAGE_SIZE = (1280, 960)
DEFAULT_MIN_DISTANCE = 25

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_MODEL_CACHE: Dict[Tuple[str, str], nn.Module] = {}


class HybridUNetResNet(nn.Module):
    def __init__(self, backbone: str = "resnet101", n_classes: int = 4):
        super().__init__()

        if backbone == "resnet18":
            resnet = models.resnet18(weights=None)
            enc_ch = [64, 64, 128, 256, 512]
        elif backbone == "resnet50":
            resnet = models.resnet50(weights=None)
            enc_ch = [64, 256, 512, 1024, 2048]
        else:  # resnet101 or default
            resnet = models.resnet101(weights=None)
            enc_ch = [64, 256, 512, 1024, 2048]

        # Encoder
        self.encoder0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.pool0 = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        # Decoder
        self.upconv4 = nn.ConvTranspose2d(enc_ch[4], enc_ch[3], 2, stride=2)
        self.dec4 = nn.Conv2d(enc_ch[3] + enc_ch[3], enc_ch[3], 3, padding=1)

        self.upconv3 = nn.ConvTranspose2d(enc_ch[3], enc_ch[2], 2, stride=2)
        self.dec3 = nn.Conv2d(enc_ch[2] + enc_ch[2], enc_ch[2], 3, padding=1)

        self.upconv2 = nn.ConvTranspose2d(enc_ch[2], enc_ch[1], 2, stride=2)
        self.dec2 = nn.Conv2d(enc_ch[1] + enc_ch[1], enc_ch[1], 3, padding=1)

        self.upconv1 = nn.ConvTranspose2d(enc_ch[1], enc_ch[0], 2, stride=2)
        self.dec1 = nn.Conv2d(enc_ch[0] + enc_ch[0], 64, 3, padding=1)

        # Single head for multi-class segmentation
        self.final = nn.Conv2d(64, n_classes, 1)
        self.n_classes = n_classes

    def forward(self, x):
        x0 = self.encoder0(x)
        x1 = self.pool0(x0)
        x2 = self.encoder1(x1)
        x3 = self.encoder2(x2)
        x4 = self.encoder3(x3)
        x5 = self.encoder4(x4)

        d4 = self.upconv4(x5)
        d4 = F.interpolate(d4, size=x4.shape[2:], mode='bilinear', align_corners=True)
        d4 = torch.cat([d4, x4], dim=1)
        d4 = self.dec4(d4)

        d3 = self.upconv3(d4)
        d3 = F.interpolate(d3, size=x3.shape[2:], mode='bilinear', align_corners=True)
        d3 = torch.cat([d3, x3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.upconv2(d3)
        d2 = F.interpolate(d2, size=x2.shape[2:], mode='bilinear', align_corners=True)
        d2 = torch.cat([d2, x2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.upconv1(d2)
        d1 = F.interpolate(d1, size=x0.shape[2:], mode='bilinear', align_corners=True)
        d1 = torch.cat([d1, x0], dim=1)
        d1 = self.dec1(d1)

        out = self.final(d1)
        out = F.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=True)
        return out


# ==========================================
# Model Abstraction Classes
# ==========================================

class ModelInference(ABC):
    """Base abstract class for model inference"""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
    
    @abstractmethod
    def load(self):
        """Load the model"""
        pass
    
    @abstractmethod
    def infer_poles(
        self,
        image: np.ndarray,
        original_size: Tuple[int, int],
        image_size: Tuple[int, int],
        confidence_threshold: float,
        iou_threshold: Optional[float],
        meta: Dict,
    ) -> List[Dict]:
        """
        Run inference and extract pole coordinates.
        
        Args:
            image: Input image as PIL Image or numpy array
            original_size: Original image size (W, H)
            image_size: Resized image size for inference (W, H)
            confidence_threshold: Confidence threshold for detections
            meta: Metadata dict with GPS and yaw info
            
        Returns:
            List of detected poles as [{"lat": float, "lon": float, "confidence": float}, ...]
        """
        pass
    
    @abstractmethod
    def get_row_geojson(self, session_dir: str, base_name: str, image_meta: Dict) -> Optional[Dict]:
        """
        Extract vine row information if available.
        
        Returns:
            GeoJSON Feature or None
        """
        pass


class ResNetModelInference(ModelInference):
    """ResNet UNet segmentation model"""
    
    def __init__(self, model_path: str, backbone: str = "resnet101"):
        super().__init__(model_path)
        self.backbone = backbone
        self.model = None
        self._pole_heatmap = None  # Store heatmap for later access
        self._row_prob_map = None  # Store row map for later access
    
    def load(self):
        """Load ResNet model"""
        if not os.path.isfile(self.model_path):
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        # Use 4 classes: background, pole, trunk, vine_row
        self.model = HybridUNetResNet(backbone=self.backbone, n_classes=4).to(DEVICE)
        state_dict = torch.load(self.model_path, map_location=DEVICE)
        
        # Try loading with strict=False to allow for minor architecture mismatches
        try:
            self.model.load_state_dict(state_dict, strict=True)
        except RuntimeError as e:
            print(f"[Model] Warning: Strict loading failed, attempting flexible loading")
            print(f"[Model] Error: {str(e)[:200]}...")
            
            # Try loading with strict=False (ignore missing/unexpected keys)
            incompatible = self.model.load_state_dict(state_dict, strict=False)
            if incompatible.missing_keys:
                print(f"[Model] Missing {len(incompatible.missing_keys)} keys (expected for different architectures)")
            if incompatible.unexpected_keys:
                print(f"[Model] Found {len(incompatible.unexpected_keys)} unexpected keys (will be ignored)")
        
        self.model.eval()
    
    def infer_poles(
        self,
        image: Union[Image.Image, np.ndarray],
        original_size: Tuple[int, int],
        image_size: Tuple[int, int],
        confidence_threshold: float,
        iou_threshold: Optional[float],
        meta: Dict,
    ) -> List[Dict]:
        """Run ResNet inference and extract poles"""
        if self.model is None:
            self.load()
        
        # Convert PIL Image to array if needed
        if isinstance(image, Image.Image):
            input_img = image.resize(image_size, Image.BILINEAR)
            image_np = np.array(input_img, dtype=np.float32) / 255.0
        else:
            image_np = cv2.resize(image, image_size)
            image_np = image_np.astype(np.float32) / 255.0
        
        # Prepare tensor
        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
        
        # Run inference - returns class probability map [batch, n_classes, height, width]
        with torch.no_grad():
            out = self.model(image_tensor)  # [1, 4, H, W]
        
        # Extract class probabilities
        out_probs = torch.softmax(out, dim=1)  # [1, 4, H, W]
        
        # Class 1 is "pole", class 3 is "vine_row"
        pole_prob = out_probs[0, 1, :, :].cpu().numpy().astype(np.float32)  # Pole class
        row_prob_map = out_probs[0, 3, :, :].cpu().numpy().astype(np.float32)  # Vine row class
        
        # Store heatmaps for later access
        self._pole_heatmap = pole_prob
        self._row_prob_map = row_prob_map
        
        # Resize to original image size
        pole_map_full = cv2.resize(pole_prob, original_size, interpolation=cv2.INTER_LINEAR)
        
        # Extract peaks
        min_distance = DEFAULT_MIN_DISTANCE
        coords = peak_local_max(pole_map_full, min_distance=min_distance, threshold_abs=confidence_threshold)
        pole_peaks = coords[:, ::-1]  # Convert from (y,x) to (x,y)
        
        # Convert to GPS
        flight_yaw = meta.get("yaw_flight", 0.0)
        gimbal_yaw = meta.get("yaw_gimbal", 0.0)
        gps_lat = meta.get("gps_lat", 0.0)
        gps_lon = meta.get("gps_lon", 0.0)
        gps_alt = meta.get("gps_alt", 0.0)
        
        poles = []
        for px, py in pole_peaks:
            try:
                lat, lon = get_gps_from_pixel(
                    int(px), int(py),
                    original_size[0], original_size[1],
                    flight_yaw, gimbal_yaw,
                    gps_lat, gps_lon, gps_alt,
                    FOCAL_LENGTH_MM, SENSOR_WIDTH_MM, SENSOR_HEIGHT_MM,
                )
                conf = float(pole_map_full[int(py), int(px)])
                poles.append({"lat": lat, "lon": lon, "confidence": conf})
            except Exception as e:
                print(f"Warning: Failed to convert pixel ({px}, {py}) to GPS: {e}")
        
        return poles
    
    def get_row_geojson(self, session_dir: str, base_name: str, image_meta: Dict) -> Optional[Dict]:
        """Extract vine rows from ResNet row map (class 3)"""
        if not hasattr(self, '_row_prob_map') or self._row_prob_map is None:
            return None
        
        row_prob_map = self._row_prob_map
        original_w, original_h = image_meta["original_size"]
        
        # Resize to original size
        row_mask_full = cv2.resize(
            (row_prob_map > 0.5).astype(np.uint8),
            (original_w, original_h),
            interpolation=cv2.INTER_NEAREST,
        )
        
        # Find contours
        contours, _ = cv2.findContours(
            row_mask_full, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        flight_yaw = image_meta["yaw"]["flight"]
        gimbal_yaw = image_meta["yaw"]["gimbal"]
        gps_lat = image_meta["gps"]["lat"]
        gps_lon = image_meta["gps"]["lon"]
        gps_alt = image_meta["gps"]["alt"]
        
        def to_gps(px: int, py: int) -> Tuple[float, float]:
            return get_gps_from_pixel(
                px, py,
                original_w, original_h,
                flight_yaw, gimbal_yaw,
                gps_lat, gps_lon, gps_alt,
                FOCAL_LENGTH_MM, SENSOR_WIDTH_MM, SENSOR_HEIGHT_MM,
            )
        
        for cnt in contours:
            if cv2.contourArea(cnt) > 2000:
                approx = cv2.approxPolyDP(cnt, 5.0, True)
                row_poly_px = approx.reshape(-1, 2)
                row_gps = []
                for px, py in row_poly_px:
                    lat, lon = to_gps(px, py)
                    row_gps.append([lon, lat])
                if len(row_gps) > 2:
                    row_gps.append(row_gps[0])
                    return {
                        "type": "Feature",
                        "geometry": {"type": "Polygon", "coordinates": [row_gps]},
                        "properties": {"image": image_meta["image_path"]},
                    }
        
        return None


class YOLOObjectDetectionInference(ModelInference):
    """YOLO object detection model (pole detection)"""
    
    def __init__(self, model_path: str):
        super().__init__(model_path)
        self.model = None
        self._yolo_device = None
    
    def load(self):
        """Load YOLO object detection model"""
        if not ULTRALYTICS_AVAILABLE:
            raise ImportError("ultralytics not available. Install with: pip install ultralytics")
        
        if not os.path.isfile(self.model_path):
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        self.model = YOLO(self.model_path)
        # Force CPU inference if requested via environment variable (avoids GPU OOM)
        yolo_device = "cpu" if os.environ.get("FORCE_CPU", "").lower() in ("1", "true", "yes") else (0 if torch.cuda.is_available() else "cpu")
        self._yolo_device = yolo_device
        self.model.to(yolo_device)
    
    def infer_poles(
        self,
        image: Union[Image.Image, np.ndarray],
        original_size: Tuple[int, int],
        image_size: Tuple[int, int],
        confidence_threshold: float,
        iou_threshold: Optional[float],
        meta: Dict,
    ) -> List[Dict]:
        """Run YOLO object detection and extract poles"""
        if self.model is None:
            self.load()
        
        # Convert PIL Image to array if needed
        if isinstance(image, Image.Image):
            # Ultralytics expects numpy images in BGR channel order.
            image_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        else:
            image_np = image
        
        # Run inference
        # YOLO will handle resizing internally
        results = self.model.predict(
            source=image_np,
            imgsz=image_size,
            conf=confidence_threshold,
            iou=0.3 if iou_threshold is None else float(iou_threshold),
            device=getattr(self, '_yolo_device', 0 if torch.cuda.is_available() else "cpu"),
            verbose=False,
        )
        
        # Extract poles from detections
        flight_yaw = meta.get("yaw_flight", 0.0)
        gimbal_yaw = meta.get("yaw_gimbal", 0.0)
        gps_lat = meta.get("gps_lat", 0.0)
        gps_lon = meta.get("gps_lon", 0.0)
        gps_alt = meta.get("gps_alt", 0.0)
        
        poles = []
        for result in results:
            if result.boxes is None:
                continue
            
            boxes = result.boxes
            for i, box in enumerate(boxes):
                # Get bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                
                # Get confidence
                conf = float(box.conf[0].cpu().numpy())
                
                # Calculate center of bounding box
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                
                try:
                    lat, lon = get_gps_from_pixel(
                        center_x, center_y,
                        original_size[0], original_size[1],
                        flight_yaw, gimbal_yaw,
                        gps_lat, gps_lon, gps_alt,
                        FOCAL_LENGTH_MM, SENSOR_WIDTH_MM, SENSOR_HEIGHT_MM,
                    )
                    poles.append({"lat": lat, "lon": lon, "confidence": conf})
                except Exception as e:
                    print(f"Warning: Failed to convert pixel ({center_x}, {center_y}) to GPS: {e}")
        
        return poles
    
    def get_row_geojson(self, session_dir: str, base_name: str, image_meta: Dict) -> Optional[Dict]:
        """YOLO object detection doesn't provide row info"""
        return None


class YOLOSegmentationInference(ModelInference):
    """YOLO segmentation model (instance segmentation)"""
    
    def __init__(self, model_path: str):
        super().__init__(model_path)
        self.model = None
        self._yolo_device = None
        self._class_map = {}

    @staticmethod
    def _normalize_class_name(name: str) -> str:
        return str(name).strip().lower().replace("-", "_").replace(" ", "_")

    @classmethod
    def _canonicalize_class_name(cls, name: str) -> str:
        clean = cls._normalize_class_name(name)
        aliases = {
            "pole": {"pole", "post", "support_post", "vine_post"},
            "trunk": {"trunk", "vine_trunk", "grape_trunk"},
            "vine_row": {"vine_row", "vine-row", "vine row", "row", "grape_row"},
        }
        for canonical_name, candidates in aliases.items():
            if clean in candidates:
                return canonical_name
        return clean

    @staticmethod
    def _polygon_centroid_xy(poly_xy: np.ndarray) -> Optional[Tuple[float, float]]:
        if poly_xy is None or len(poly_xy) < 3:
            return None
        x = poly_xy[:, 0].astype(np.float64)
        y = poly_xy[:, 1].astype(np.float64)
        x_next = np.roll(x, -1)
        y_next = np.roll(y, -1)
        cross = x * y_next - x_next * y
        area2 = float(np.sum(cross))
        if abs(area2) < 1e-10:
            return float(np.mean(x)), float(np.mean(y))
        cx = float(np.sum((x + x_next) * cross) / (3.0 * area2))
        cy = float(np.sum((y + y_next) * cross) / (3.0 * area2))
        return cx, cy
    
    def load(self):
        """Load YOLO segmentation model"""
        if not ULTRALYTICS_AVAILABLE:
            raise ImportError("ultralytics not available. Install with: pip install ultralytics")
        
        if not os.path.isfile(self.model_path):
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        self.model = YOLO(self.model_path)
        names = getattr(self.model, "names", None)
        if names is None:
            names = getattr(getattr(self.model, "model", None), "names", None)
        class_map = {}
        if isinstance(names, dict):
            for class_id, class_name in names.items():
                class_map[int(class_id)] = self._canonicalize_class_name(class_name)
        elif isinstance(names, list):
            for class_id, class_name in enumerate(names):
                class_map[class_id] = self._canonicalize_class_name(class_name)
        self._class_map = class_map
        # Force CPU inference if requested via environment variable (avoids GPU OOM)
        yolo_device = "cpu" if os.environ.get("FORCE_CPU", "").lower() in ("1", "true", "yes") else (0 if torch.cuda.is_available() else "cpu")
        self._yolo_device = yolo_device
        self.model.to(yolo_device)
    
    def infer_poles(
        self,
        image: Union[Image.Image, np.ndarray],
        original_size: Tuple[int, int],
        image_size: Tuple[int, int],
        confidence_threshold: float,
        iou_threshold: Optional[float],
        meta: Dict,
    ) -> List[Dict]:
        """Run YOLO segmentation and extract poles"""
        if self.model is None:
            self.load()
        
        # Convert PIL Image to array if needed
        if isinstance(image, Image.Image):
            # Ultralytics expects numpy images in BGR channel order.
            image_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        else:
            image_np = image
        
        # Run inference
        results = self.model.predict(
            source=image_np,
            imgsz=image_size,
            conf=confidence_threshold,
            iou=0.3 if iou_threshold is None else float(iou_threshold),
            device=getattr(self, '_yolo_device', 0 if torch.cuda.is_available() else "cpu"),
            verbose=False,
        )
        
        # Extract poles from segmentations
        flight_yaw = meta.get("yaw_flight", 0.0)
        gimbal_yaw = meta.get("yaw_gimbal", 0.0)
        gps_lat = meta.get("gps_lat", 0.0)
        gps_lon = meta.get("gps_lon", 0.0)
        gps_alt = meta.get("gps_alt", 0.0)
        
        poles = []
        for result in results:
            if result.boxes is None or len(result.boxes) == 0:
                continue

            boxes = result.boxes
            masks_xy = []
            if getattr(result, "masks", None) is not None and result.masks.xy is not None:
                masks_xy = result.masks.xy

            for idx, box in enumerate(boxes):
                # Get confidence
                conf = float(box.conf[0].cpu().numpy())

                cls_id = int(box.cls[0].cpu().numpy()) if box.cls is not None else -1
                class_name = self._class_map.get(cls_id, f"class_{cls_id}")
                if class_name != "pole":
                    continue

                center_x = None
                center_y = None
                if idx < len(masks_xy):
                    centroid = self._polygon_centroid_xy(np.asarray(masks_xy[idx]))
                    if centroid is not None:
                        center_x, center_y = centroid

                if center_x is None or center_y is None:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    center_x = float((x1 + x2) / 2.0)
                    center_y = float((y1 + y2) / 2.0)
                
                try:
                    lat, lon = get_gps_from_pixel(
                        int(round(center_x)), int(round(center_y)),
                        original_size[0], original_size[1],
                        flight_yaw, gimbal_yaw,
                        gps_lat, gps_lon, gps_alt,
                        FOCAL_LENGTH_MM, SENSOR_WIDTH_MM, SENSOR_HEIGHT_MM,
                    )
                    poles.append({"lat": lat, "lon": lon, "confidence": conf})
                except Exception as e:
                    print(f"Warning: Failed to convert pixel ({center_x}, {center_y}) to GPS: {e}")
        
        return poles
    
    def get_row_geojson(self, session_dir: str, base_name: str, image_meta: Dict) -> Optional[Dict]:
        """YOLO segmentation doesn't provide row info"""
        return None


def _detect_model_type(model_path: str) -> str:
    """
    Detect model type based on file extension and directory structure.
    Returns: "resnet", "yolo_object", or "yolo_segmentation"
    """
    model_path_lower = model_path.lower()
    
    if model_path_lower.endswith(".pth"):
        # Check if it's in resnet directory
        if "resnet" in model_path_lower:
            return "resnet"
        # Check if it's in yolo_object_detection_models directory
        if "yolo_object" in model_path_lower or "yolo_detection" in model_path_lower:
            return "yolo_object"
        # Check if it's in yolo_segmentation_models directory
        if "yolo_segment" in model_path_lower or "yolo_seg" in model_path_lower:
            return "yolo_segmentation"
        # Default to resnet for .pth files
        return "resnet"
    
    elif model_path_lower.endswith(".pt"):
        # YOLO models are .pt files
        # Check if it's in yolo_segmentation_models directory
        if "yolo_segment" in model_path_lower or "yolo_seg" in model_path_lower:
            return "yolo_segmentation"
        # Default to object detection for other .pt files
        return "yolo_object"
    
    # Default fallback
    return "resnet"


def _detect_resnet_backbone(model_path: str) -> str:
    """
    Detect ResNet backbone (18, 50, or 101) from model path.
    Returns: "resnet18", "resnet50", or "resnet101" (default)
    """
    model_path_lower = model_path.lower()
    
    if "resnet18" in model_path_lower:
        return "resnet18"
    elif "resnet50" in model_path_lower:
        return "resnet50"
    elif "resnet101" in model_path_lower:
        return "resnet101"
    
    # Default to resnet101
    return "resnet101"


def _load_model(model_path: str, backbone: str = "resnet101") -> ModelInference:
    """Load a model and return appropriate inference wrapper"""
    model_type = _detect_model_type(model_path)
    
    print(f"[Model] Loading {model_type} model: {model_path}")
    
    if model_type == "resnet":
        # Auto-detect backbone from path
        detected_backbone = _detect_resnet_backbone(model_path)
        model = ResNetModelInference(model_path, backbone=detected_backbone)
    elif model_type == "yolo_object":
        model = YOLOObjectDetectionInference(model_path)
    elif model_type == "yolo_segmentation":
        model = YOLOSegmentationInference(model_path)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model.load()
    return model


def _old_load_model(model_path: str, backbone: str) -> nn.Module:
    """Legacy ResNet model loader - kept for compatibility"""
    cache_key = (model_path, backbone)
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = HybridUNetResNet(backbone=backbone).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    _MODEL_CACHE[cache_key] = model
    return model


def _resize_heatmap(heatmap: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    return cv2.resize(heatmap, target_shape, interpolation=cv2.INTER_LINEAR)


def _get_peak_coordinates(heatmap: np.ndarray, threshold: float, min_distance: int) -> np.ndarray:
    coords = peak_local_max(heatmap, min_distance=min_distance, threshold_abs=threshold)
    return coords[:, ::-1]


def _haversine_distance_m(lat1, lon1, lat2, lon2) -> float:
    r = 6371000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return r * c


def _pairwise_haversine_m(coords_latlon: np.ndarray) -> np.ndarray:
    n = coords_latlon.shape[0]
    dists = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        lat1, lon1 = coords_latlon[i]
        for j in range(i + 1, n):
            lat2, lon2 = coords_latlon[j]
            d = _haversine_distance_m(lat1, lon1, lat2, lon2)
            dists[i, j] = d
            dists[j, i] = d
    return dists


def _to_local_meters(points: List[Dict[str, float]]) -> Tuple[np.ndarray, Tuple[float, float]]:
    lats = np.array([p["lat"] for p in points], dtype=np.float64)
    lons = np.array([p["lon"] for p in points], dtype=np.float64)
    center_lat = float(np.mean(lats))
    center_lon = float(np.mean(lons))
    lat_to_m = 111111.0
    lon_to_m = 111111.0 * np.cos(np.radians(center_lat))
    x = (lons - center_lon) * lon_to_m
    y = (lats - center_lat) * lat_to_m
    return np.column_stack([x, y]), (center_lat, center_lon)


def _latlon_to_xy(lat: float, lon: float, ref_lat: float) -> Tuple[float, float]:
    meters_per_deg_lat = 111320.0
    meters_per_deg_lon = 111320.0 * math.cos(math.radians(ref_lat))
    x = lon * meters_per_deg_lon
    y = lat * meters_per_deg_lat
    return x, y


def _point_segment_distance_m(
    point: Tuple[float, float],
    a: Tuple[float, float],
    b: Tuple[float, float],
    ref_lat: float,
) -> float:
    px, py = _latlon_to_xy(point[0], point[1], ref_lat)
    ax, ay = _latlon_to_xy(a[0], a[1], ref_lat)
    bx, by = _latlon_to_xy(b[0], b[1], ref_lat)

    vx = bx - ax
    vy = by - ay
    wx = px - ax
    wy = py - ay

    vv = vx * vx + vy * vy
    if vv == 0.0:
        return math.hypot(px - ax, py - ay)

    t = (wx * vx + wy * vy) / vv
    t = max(0.0, min(1.0, t))
    projx = ax + t * vx
    projy = ay + t * vy
    return math.hypot(px - projx, py - projy)


def _point_in_ring(point: Tuple[float, float], ring: np.ndarray) -> bool:
    lat, lon = point
    inside = False
    n = ring.shape[0]
    for i in range(n):
        j = (i + 1) % n
        yi, xi = ring[i, 0], ring[i, 1]
        yj, xj = ring[j, 0], ring[j, 1]
        intersect = ((xi > lon) != (xj > lon)) and (
            lat < (yj - yi) * (lon - xi) / (xj - xi + 1e-16) + yi
        )
        if intersect:
            inside = not inside
    return inside


def _min_distance_to_rows_m(
    point: Tuple[float, float],
    rows: List[np.ndarray],
    polygons: List[List[np.ndarray]],
    ref_lat: float,
) -> float:
    for poly in polygons:
        if not poly:
            continue
        if _point_in_ring(point, poly[0]):
            return 0.0

    best = float("inf")
    for row in rows:
        if row.shape[0] == 0:
            continue
        for i in range(row.shape[0] - 1):
            a = (float(row[i, 0]), float(row[i, 1]))
            b = (float(row[i + 1, 0]), float(row[i + 1, 1]))
            d = _point_segment_distance_m(point, a, b, ref_lat)
            if d < best:
                best = d
                if best == 0.0:
                    return 0.0

    for poly in polygons:
        for ring in poly:
            if ring.shape[0] == 0:
                continue
            for i in range(ring.shape[0] - 1):
                a = (float(ring[i, 0]), float(ring[i, 1]))
                b = (float(ring[i + 1, 0]), float(ring[i + 1, 1]))
                d = _point_segment_distance_m(point, a, b, ref_lat)
                if d < best:
                    best = d
                    if best == 0.0:
                        return 0.0

    return best


def _filter_poles_by_vine_rows(
    poles: List[Dict[str, float]],
    vine_rows_geojson: Dict,
    max_distance_m: float = 0.5,
) -> List[Dict[str, float]]:
    """
    Filter poles to keep only those within vine row polygons or within max_distance_m of them.
    
    Args:
        poles: List of pole dictionaries with 'lat', 'lon' keys
        vine_rows_geojson: GeoJSON FeatureCollection of vine row polygons
        max_distance_m: Maximum distance in meters from vine rows to keep poles (default 0.5m)
    
    Returns:
        Filtered list of poles
    """
    print(f"[vine-row-filter] start poles={len(poles)} max_distance_m={max_distance_m}")

    if not poles:
        return []
    
    features = vine_rows_geojson.get("features", [])
    if not features:
        # No vine rows detected, return all poles
        print("[vine-row-filter] no vine row polygons; keeping all poles")
        return poles
    
    rows: List[np.ndarray] = []
    polygons: List[List[np.ndarray]] = []
    for feat in features:
        geom = feat.get("geometry", {})
        gtype = geom.get("type")
        coords = geom.get("coordinates")
        if coords is None:
            continue
        if gtype == "LineString":
            rows.append(np.array([[c[1], c[0]] for c in coords], dtype=np.float64))
        elif gtype == "MultiLineString":
            for line in coords:
                rows.append(np.array([[c[1], c[0]] for c in line], dtype=np.float64))
        elif gtype == "Polygon":
            rings = [np.array([[c[1], c[0]] for c in ring], dtype=np.float64) for ring in coords]
            polygons.append(rings)
        elif gtype == "MultiPolygon":
            for poly in coords:
                rings = [np.array([[c[1], c[0]] for c in ring], dtype=np.float64) for ring in poly]
                polygons.append(rings)

    if not rows and not polygons:
        print("[vine-row-filter] no line/polygon features; keeping all poles")
        return poles

    all_lats = []
    for row in rows:
        if row.size:
            all_lats.extend(row[:, 0].tolist())
    for poly in polygons:
        for ring in poly:
            if ring.size:
                all_lats.extend(ring[:, 0].tolist())

    ref_lat = float(np.mean(all_lats)) if all_lats else float(np.mean([p["lat"] for p in poles]))

    filtered_poles = []
    rejected = 0
    for pole in poles:
        point = (pole["lat"], pole["lon"])
        d = _min_distance_to_rows_m(point, rows, polygons, ref_lat)
        if d <= max_distance_m:
            filtered_poles.append(pole)
        else:
            rejected += 1

    print(f"[vine-row-filter] kept={len(filtered_poles)} rejected={rejected} rows={len(rows)} polygons={len(polygons)}")
    return filtered_poles


def _estimate_k(points_xy: np.ndarray, eps_m: float) -> int:
    n_points = points_xy.shape[0]
    if n_points <= 1:
        return 1
    min_x, min_y = np.min(points_xy, axis=0)
    max_x, max_y = np.max(points_xy, axis=0)
    area = max(0.0, (max_x - min_x) * (max_y - min_y))
    if area <= 0 or eps_m <= 0:
        return min(1, n_points)
    approx_cluster_area = math.pi * (eps_m ** 2)
    k = int(max(1.0, area / approx_cluster_area))
    return max(1, min(n_points, k))


def _cluster_poles(
    points: List[Dict[str, float]],
    eps_m: float,
    algorithm: str,
    cluster_params: Optional[Dict] = None,
) -> List[Dict[str, float]]:
    if not points:
        return []

    algo = (algorithm or "dbscan").lower()
    params = cluster_params or {}
    if algo in ("none", "no_clustering", "raw"):
        return [
            {
                "lat": p["lat"],
                "lon": p["lon"],
                "confidence": p["confidence"],
                "count": 1,
                "cluster_id": idx,
            }
            for idx, p in enumerate(points)
        ]

    if algo == "dbscan":
        coords = np.array([[p["lat"], p["lon"]] for p in points], dtype=np.float64)
        dists = _pairwise_haversine_m(coords)
        min_samples = int(params.get("dbscan_min_samples", 3))
        min_samples = max(1, min_samples)
        labels = DBSCAN(eps=eps_m, min_samples=min_samples, metric="precomputed").fit_predict(dists)
    elif algo in ("agglomerative", "average"):
        coords = np.array([[p["lat"], p["lon"]] for p in points], dtype=np.float64)
        dists = _pairwise_haversine_m(coords)
        try:
            clustering = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=eps_m,
                linkage="average",
                metric="precomputed",
            )
        except TypeError:
            clustering = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=eps_m,
                linkage="average",
                affinity="precomputed",
            )
        labels = clustering.fit_predict(dists)
    elif algo == "kmeans":
        xy, _center = _to_local_meters(points)
        k_override = params.get("kmeans_k", None)
        if k_override is not None:
            try:
                k = int(k_override)
            except (TypeError, ValueError):
                k = _estimate_k(xy, eps_m)
            k = max(1, min(len(xy), k))
        else:
            k = _estimate_k(xy, eps_m)
        labels = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(xy)
    elif algo == "hdbscan":
        try:
            import hdbscan
        except ImportError as exc:
            raise ImportError("hdbscan is not installed. Run `pip install hdbscan`. ") from exc
        coords = np.array([[p["lat"], p["lon"]] for p in points], dtype=np.float64)
        dists = _pairwise_haversine_m(coords)
        min_cluster_size = max(2, int(params.get("hdbscan_min_cluster_size", 2)))
        min_samples = max(1, int(params.get("hdbscan_min_samples", 2)))
        labels = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric="precomputed",
        ).fit_predict(dists)
    else:
        raise ValueError(f"Unsupported clustering algorithm: {algorithm}")

    clusters: Dict[int, List[Dict[str, float]]] = {}
    for point, label in zip(points, labels):
        clusters.setdefault(int(label), []).append(point)

    merged = []
    for label, cluster_points in clusters.items():
        lat = float(np.mean([p["lat"] for p in cluster_points]))
        lon = float(np.mean([p["lon"] for p in cluster_points]))
        conf = float(np.max([p["confidence"] for p in cluster_points]))
        merged.append({
            "lat": lat,
            "lon": lon,
            "confidence": conf,
            "count": len(cluster_points),
            "cluster_id": label,
        })
    return merged


def _build_geojson(points: List[Dict[str, float]]) -> Dict:
    features = []
    for idx, p in enumerate(points):
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [p["lon"], p["lat"]]},
            "properties": {
                "confidence": p["confidence"],
                "count": p.get("count", 1),
                "id": f"pole_{idx}",
            },
        })
    return {"type": "FeatureCollection", "features": features}


def _build_gps_converter(meta: Dict) -> Tuple[float, float, float, float, float, float]:
    flight_yaw = meta["yaw"]["flight"]
    gimbal_yaw = meta["yaw"]["gimbal"]
    gps_lat = meta["gps"]["lat"]
    gps_lon = meta["gps"]["lon"]
    gps_alt = meta["gps"]["alt"]
    return flight_yaw, gimbal_yaw, gps_lat, gps_lon, gps_alt


def run_inference(
    input_dir: str,
    model_path: str,
    cache_root: str,
    image_size: Optional[Tuple[int, int]] = None,
    backbone: str = "resnet101",
    confidence_threshold: float = 0.4,
    iou_threshold: Optional[float] = None,
    cluster_eps_m: float = 1.5,
    cluster_algo: str = "dbscan",
    min_distance: int = DEFAULT_MIN_DISTANCE,
    filter_by_vine_rows: bool = False,
    progress_callback=None,
) -> Tuple[str, Dict, Dict, Dict]:
    """
    Run inference on a directory of images using the specified model.
    
    Supports multiple model types (ResNet, YOLO object detection, YOLO segmentation).
    """
    model = _load_model(model_path, backbone)

    # Determine appropriate image size for inference.
    # If the caller did not explicitly set an image size, prefer the model's trained size (e.g. YOLO models).
    if image_size is None:
        image_size = DEFAULT_IMAGE_SIZE
    elif isinstance(image_size, list):
        image_size = tuple(image_size)

    # If using a YOLO model and the caller passed the default size, use the model's trained image size.
    # This avoids missing detections when the model expects a larger input resolution.
    if isinstance(model, (YOLOObjectDetectionInference, YOLOSegmentationInference)):
        try:
            yolo_imgsz = getattr(model.model.args, "imgsz", None)
            if yolo_imgsz:
                # If caller used default image size, or passed a smaller size, prefer the model's size.
                if (
                    image_size == DEFAULT_IMAGE_SIZE
                    or (isinstance(image_size, tuple) and (image_size[0] < yolo_imgsz or image_size[1] < yolo_imgsz))
                ):
                    image_size = (int(yolo_imgsz), int(yolo_imgsz))
        except Exception:
            pass

    # Default IoU threshold for YOLO models to match evaluation setup.
    effective_iou_threshold = iou_threshold
    if isinstance(model, (YOLOObjectDetectionInference, YOLOSegmentationInference)) and effective_iou_threshold is None:
        effective_iou_threshold = 0.3

    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"Image folder not found: {input_dir}")

    session_id = uuid.uuid4().hex
    session_dir = os.path.join(cache_root, session_id)
    os.makedirs(session_dir, exist_ok=True)

    image_files = [
        f for f in os.listdir(input_dir)
        if f.lower().endswith(IMAGE_EXTS)
    ]
    image_files.sort()
    total_images = len(image_files)

    if progress_callback:
        progress_callback(0, max(total_images, 1), "Preparing images")

    all_points: List[Dict[str, float]] = []
    meta_images = []

    for idx, filename in enumerate(image_files):
        img_path = os.path.join(input_dir, filename)
        image_pil = Image.open(img_path).convert("RGB")
        original_w, original_h = image_pil.size

        exif = extract_exif(img_path)
        (
            flight_yaw,
            _flight_pitch,
            _flight_roll,
            gimbal_yaw,
            _gimbal_pitch,
            _gimbal_roll,
            gps_lat,
            gps_lon,
            gps_alt,
            _fov,
            _,
            _,
            _,
        ) = exif

        if gps_lat is None:
            continue

        flight_yaw_num = extract_number(flight_yaw)
        gimbal_yaw_num = extract_number(gimbal_yaw)
        if not gimbal_yaw_num:
            gimbal_yaw_num = flight_yaw_num
        gps_alt_num = extract_number(gps_alt) if gps_alt else 0.0

        # Run model inference
        meta_dict = {
            "yaw_flight": float(flight_yaw_num or 0.0),
            "yaw_gimbal": float(gimbal_yaw_num or 0.0),
            "gps_lat": float(gps_lat),
            "gps_lon": float(gps_lon),
            "gps_alt": float(gps_alt_num),
        }
        
        poles = model.infer_poles(
            image_pil,
            (original_w, original_h),
            image_size,
            confidence_threshold,
            effective_iou_threshold,
            meta_dict,
        )
        all_points.extend(poles)

        # Save metadata and pole detections for later use
        base_name = os.path.splitext(filename)[0]
        npz_path = os.path.join(session_dir, f"{base_name}_outputs.npz")
        
        # For ResNet models, save heatmaps in old format for filtering compatibility
        # For YOLO models, save poles directly
        if isinstance(model, ResNetModelInference):
            pole_heatmap = model._pole_heatmap
            row_prob_map = model._row_prob_map if hasattr(model, '_row_prob_map') else None
            if row_prob_map is not None:
                np.savez_compressed(npz_path, pole=pole_heatmap, row=row_prob_map)
            else:
                np.savez_compressed(npz_path, pole=pole_heatmap, poles_cached=np.array([[p["lat"], p["lon"], p["confidence"]] for p in poles]) if poles else np.array([]))
        else:
            # YOLO models: save poles directly since we can't re-extract from confidence threshold
            poles_array = np.array([[p["lat"], p["lon"], p["confidence"]] for p in poles]) if poles else np.array([])
            np.savez_compressed(npz_path, poles=poles_array)

        image_meta = {
            "image_path": img_path,
            "npz_path": npz_path,
            "original_size": [original_w, original_h],
            "gps": {"lat": float(gps_lat), "lon": float(gps_lon), "alt": float(gps_alt_num)},
            "yaw": {"flight": float(flight_yaw_num or 0.0), "gimbal": float(gimbal_yaw_num or 0.0)},
            "camera": {
                "focal_length_mm": FOCAL_LENGTH_MM,
                "sensor_width_mm": SENSOR_WIDTH_MM,
                "sensor_height_mm": SENSOR_HEIGHT_MM,
            },
        }
        meta_images.append(image_meta)

        if progress_callback:
            progress_callback(idx + 1, max(total_images, 1), f"Processed {idx + 1}/{max(total_images, 1)}")

    raw_points_count = len(all_points)
    poles_geojson = _build_geojson(all_points)

    # Extract vine rows if supported by the model
    vine_rows_geojson = {"type": "FeatureCollection", "features": []}
    if isinstance(model, ResNetModelInference):
        # Only ResNet model provides row information
        for idx, filename in enumerate(image_files):
            if idx >= len(meta_images):
                break
            image_meta = meta_images[idx]
            img_path = os.path.join(input_dir, filename)
            image_pil = Image.open(img_path).convert("RGB")
            original_w, original_h = image_pil.size
            
            # Re-run inference to extract row geojson
            meta_dict = {
                "yaw_flight": image_meta["yaw"]["flight"],
                "yaw_gimbal": image_meta["yaw"]["gimbal"],
                "gps_lat": image_meta["gps"]["lat"],
                "gps_lon": image_meta["gps"]["lon"],
                "gps_alt": image_meta["gps"]["alt"],
            }
            row_feature = model.get_row_geojson(session_dir, os.path.splitext(filename)[0], image_meta)
            if row_feature:
                vine_rows_geojson["features"].append(row_feature)

    filtered_points = all_points
    filter_elapsed_s = 0.0
    if filter_by_vine_rows:
        if progress_callback:
            progress_callback(max(total_images, 1), max(total_images, 1), "Filtering poles by vine rows")
        filter_start = time.perf_counter()
        filtered_points = _filter_poles_by_vine_rows(all_points, vine_rows_geojson, max_distance_m=0.5)
        filter_elapsed_s = time.perf_counter() - filter_start
        if progress_callback:
            progress_callback(max(total_images, 1), max(total_images, 1), "Filtering complete")

    filtered_count = len(filtered_points)
    poles_geojson = _build_geojson(filtered_points)

    stats = {
        "images_total": total_images,
        "images_with_gps": len(meta_images),
        "raw_poles": raw_points_count,
        "filtered_poles": filtered_count,
        "vine_row_filtering": bool(filter_by_vine_rows),
        "filter_elapsed_s": round(filter_elapsed_s, 3),
        "clustered_poles": None,
        "cluster_eps_m": cluster_eps_m,
        "cluster_algo": cluster_algo,
        "confidence_threshold": confidence_threshold,
        "iou_threshold": effective_iou_threshold,
        "min_distance_px": min_distance,
        "model_type": _detect_model_type(model_path),
    }

    meta = {
        "session_id": session_id,
        "created_at": datetime.datetime.utcnow().isoformat() + "Z",
        "image_size": [image_size[0], image_size[1]],
        "images": meta_images,
        "vine_rows_geojson": vine_rows_geojson,
        "model_type": _detect_model_type(model_path),
        "model_path": model_path,
    }
    meta_path = os.path.join(cache_root, f"{session_id}.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return session_id, poles_geojson, stats, vine_rows_geojson


def load_session_meta(cache_root: str, session_id: str) -> Dict:
    meta_path = os.path.join(cache_root, f"{session_id}.json")
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(f"Session metadata not found: {session_id}")
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def apply_filter(
    cache_root: str,
    session_id: str,
    confidence_threshold: float,
    cluster_eps_m: float,
    cluster_algo: str = "dbscan",
    cluster_params: Optional[Dict] = None,
    min_distance: int = DEFAULT_MIN_DISTANCE,
    rows_geojson: Dict | None = None,
    filter_by_vine_rows: bool = True,
    progress_callback=None,
) -> Dict:
    meta = load_session_meta(cache_root, session_id)
    all_points: List[Dict[str, float]] = []

    images = meta.get("images", [])
    total_images = len(images)
    if progress_callback:
        progress_callback(0, max(total_images, 1), "Loading cached detections")

    for idx, image_meta in enumerate(images):
        npz_path = image_meta["npz_path"]
        if not os.path.isfile(npz_path):
            continue

        data = np.load(npz_path)
        
        # Handle both old format (with "pole" heatmap) and new format (with "poles" array)
        if "pole" in data:
            # Old format: heatmap - re-extract poles with new confidence threshold
            pole_map_lowres = data["pole"]
            original_w, original_h = image_meta["original_size"]

            pole_map_full = _resize_heatmap(pole_map_lowres, (original_w, original_h))
            pole_peaks = _get_peak_coordinates(pole_map_full, confidence_threshold, min_distance=min_distance)

            flight_yaw, gimbal_yaw, gps_lat, gps_lon, gps_alt = _build_gps_converter(image_meta)

            def to_gps(px: int, py: int) -> Tuple[float, float]:
                return get_gps_from_pixel(
                    px,
                    py,
                    original_w,
                    original_h,
                    flight_yaw,
                    gimbal_yaw,
                    gps_lat,
                    gps_lon,
                    gps_alt,
                    FOCAL_LENGTH_MM,
                    SENSOR_WIDTH_MM,
                    SENSOR_HEIGHT_MM,
                )

            for px, py in pole_peaks:
                lat, lon = to_gps(px, py)
                conf = float(pole_map_full[int(py), int(px)])
                all_points.append({"lat": lat, "lon": lon, "confidence": conf})
        else:
            # New format: poles array - use cached poles directly
            # (YOLO models don't have heatmaps, so we can't re-extract)
            if "poles" in data:
                poles_array = data["poles"]
            elif "poles_cached" in data:
                poles_array = data["poles_cached"]
            else:
                poles_array = np.array([])
            
            for pole in poles_array:
                if len(pole) >= 3:
                    all_points.append({
                        "lat": float(pole[0]),
                        "lon": float(pole[1]),
                        "confidence": float(pole[2])
                    })

        if progress_callback:
            progress_callback(idx + 1, max(total_images, 1), f"Processed {idx + 1}/{max(total_images, 1)}")

    raw_points_count = len(all_points)
    filtered_points = all_points
    filter_elapsed_s = 0.0
    if filter_by_vine_rows:
        meta_rows_geojson = meta.get("vine_rows_geojson", {"type": "FeatureCollection", "features": []})
        if rows_geojson:
            merged_features = list(meta_rows_geojson.get("features", [])) + list(rows_geojson.get("features", []))
            vine_rows_geojson = {"type": "FeatureCollection", "features": merged_features}
        else:
            vine_rows_geojson = meta_rows_geojson
        if progress_callback:
            progress_callback(max(total_images, 1), max(total_images, 1), "Filtering poles by vine rows")
        filter_start = time.perf_counter()
        filtered_points = _filter_poles_by_vine_rows(all_points, vine_rows_geojson, max_distance_m=0.5)
        filter_elapsed_s = time.perf_counter() - filter_start
        if progress_callback:
            progress_callback(max(total_images, 1), max(total_images, 1), "Filtering complete")

    if progress_callback:
        progress_callback(max(total_images, 1), max(total_images, 1), "Clustering poles")

    clustered = _cluster_poles(
        filtered_points,
        eps_m=cluster_eps_m,
        algorithm=cluster_algo,
        cluster_params=cluster_params,
    )
    poles_geojson = _build_geojson(clustered)

    stats = {
        "images_total": total_images,
        "images_with_gps": len(images),
        "raw_poles": raw_points_count,
        "clustered_poles": len(clustered),
        "filtered_poles": len(filtered_points),
        "vine_row_filtering": bool(filter_by_vine_rows),
        "filter_elapsed_s": round(filter_elapsed_s, 3),
        "cluster_eps_m": cluster_eps_m,
        "cluster_algo": cluster_algo,
        "cluster_params": cluster_params or {},
        "confidence_threshold": confidence_threshold,
        "min_distance_px": min_distance,
    }

    return {"poles": poles_geojson, "stats": stats}
