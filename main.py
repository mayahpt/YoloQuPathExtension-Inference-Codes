import argparse
import logging
import os
import re
import time
from pathlib import Path

import cv2
import geojson
import numpy as np
from ultralytics import YOLO
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
from shapelysmooth import chaikin_smooth

from utils import check_version, format_shapes, read_config

os.environ["YOLO_VERBOSE"] = "False"

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)

def add_offsets(contour: list, offset_x: float, offset_y: float):
    np_contour = np.array(contour, dtype=np.float64)
    if np_contour.ndim != 2 or np_contour.shape[1] != 2:
        logging.warning(f"Invalid contour shape: {np_contour.shape}, skipping offset.")
        return []
    np_contour[:, 0] += offset_x
    np_contour[:, 1] += offset_y
    return np_contour.tolist()

def extract_tile_info(tile_name: str):
    pattern = r"\[(.*?)\]"
    matches = re.findall(pattern, tile_name)
    if matches:
        matched_string = matches[0]
        split_string = matched_string.split(",")
        extracted_dict = {"file_name": tile_name}
        for item in split_string:
            key, value = item.split("=")
            extracted_dict[key] = float(value) if key == "ds" else int(value)
        return extracted_dict
    return None

def fill_holes(geometry: Polygon | MultiPolygon, min_hole_area: float = 0.0):
    def remove_holes(p):
        new_interiors = [hole for hole in p.interiors if Polygon(hole).area > min_hole_area]
        return Polygon(p.exterior, new_interiors)

    if geometry.is_empty:
        return geometry
    if isinstance(geometry, Polygon):
        return remove_holes(geometry)
    elif isinstance(geometry, MultiPolygon):
        return MultiPolygon([remove_holes(poly) for poly in geometry.geoms if not poly.is_empty])
    return geometry

def chaikin_smooth_coords(coords, iterations=2):
    coords = np.array(coords)
    for _ in range(iterations):
        new_coords = []
        for i in range(len(coords) - 1):
            p0 = coords[i]
            p1 = coords[i + 1]
            q = 0.75 * p0 + 0.25 * p1
            r = 0.25 * p0 + 0.75 * p1
            new_coords.extend([q, r])
        # close the shape
        q = 0.75 * coords[-1] + 0.25 * coords[0]
        r = 0.25 * coords[-1] + 0.75 * coords[0]
        new_coords.extend([q, r])
        coords = np.array(new_coords)
    return coords

def smooth_polygon(polygon: Polygon, iterations: int = 2) -> Polygon:
    if polygon.is_empty or not polygon.is_valid:
        return polygon
    exterior = chaikin_smooth_coords(polygon.exterior.coords, iterations)
    return Polygon(exterior)

def xy_to_geojson(polygons: list, output_path: str, properties: dict):
    features = []
    for coords in polygons:
        if len(coords) < 3:
            continue
        if not np.all(coords[0] == coords[-1]):
            coords = np.vstack([coords, coords[0]])
        polygon = Polygon(coords)
        if polygon.is_empty or not polygon.is_valid:
            continue
        polygon = fill_holes(polygon, min_hole_area=0.0)
        polygon = polygon.simplify(tolerance=0.5, preserve_topology=True)
        polygon = smooth_polygon(polygon, iterations=3)
        #polygon = polygon.simplify(tolerance=1.0, preserve_topology=True)
        if not polygon.is_valid or polygon.is_empty:
            continue
        features.append(geojson.Feature(geometry=polygon, properties=properties))
    feature_collection = geojson.FeatureCollection(features)
    logging.info(f"Exporting {len(features)} vector objects to GeoJSON.")
    with open(output_path, "w") as f:
        f.write(geojson.dumps(feature_collection, indent=2))

def predict(config: dict, args: argparse.Namespace, operation: str, k_size: int):
    logging.info("Vector-based prediction started.")
    model = YOLO(config["model_path"])
    logging.info(f"Model args - conf: {args.conf}, iou: {args.iou}, imgsz: {args.imgsz}")

    preds = model.predict(
        config["roi_tiles_path"],
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        stream=True,
        verbose=False,
        retina_masks=True,
    )
    logging.info("Inference done. Extracting polygons...")

    raw_geoms = []
    erosion_buffer = -k_size

    for pred in preds:
        tile_info = extract_tile_info(Path(pred.path).stem)
        if tile_info is None or pred.masks is None or pred.masks.xy is None:
            continue

        for xy in pred.masks.xy:
            if len(xy) < 3:
                continue
            coords = xy * args.ds
            coords[:, 0] += tile_info["x"]
            coords[:, 1] += tile_info["y"]
            poly = Polygon(coords).buffer(erosion_buffer)
            if poly.is_empty or not poly.is_valid:
                continue
            raw_geoms.append(poly)

    logging.info(f"Merging {len(raw_geoms)} predicted shapes...")
    merged = unary_union(raw_geoms)

    if isinstance(merged, MultiPolygon):
        polygons = [np.array(poly.exterior.coords) for poly in merged.geoms if poly.is_valid and not poly.is_empty]
    elif isinstance(merged, Polygon):
        polygons = [np.array(merged.exterior.coords)]
    else:
        polygons = []

    xy_to_geojson(polygons, config["preds_path"], config["geojson_properties"])

def load_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("base_path", type=Path)
    parser.add_argument("roi_x", type=float)
    parser.add_argument("roi_y", type=float)
    parser.add_argument("roi_width", type=float)
    parser.add_argument("roi_height", type=float)
    parser.add_argument("imgsz", type=int)
    parser.add_argument("ds", type=float)
    parser.add_argument("conf", type=float)
    parser.add_argument("iou", type=float)
    return parser.parse_args()

def main():
    if not check_version():
        logging.error("PYTHON VERSION SHOULD BE AT LEAST 3.8!")
        return
    logging.info("Process started.")
    start_time = time.time()
    args = load_arguments()
    config_path = args.base_path / "config.json"
    config = read_config(config_path, args.base_path, logging.getLogger(__name__))
    predict(config, args, config["morphology"]["operation"], config["morphology"]["kernel_size"])
    logging.info(f"Process finished in {time.time() - start_time:.2f} seconds.")

if __name__ == "__main__":
    main()
