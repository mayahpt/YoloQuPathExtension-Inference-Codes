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
from shapely.geometry import Polygon,MultiPolygon

from utils import check_version, format_shapes, read_config

os.environ["YOLO_VERBOSE"] = "False"

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)

def add_offsets(contour: list, offset_x: float, offset_y: float):
    """
    Add ROI offset values to coordinates.
    """
    np_contour = np.array(contour,dtype=np.float64)

    if np_contour.ndim != 2 or np_contour.shape[1] != 2:
        logging.warning(f"Invalid contour shape: {np_contour.shape}, skipping offset.")
        return []

    np_contour[:, 0] += offset_x
    np_contour[:, 1] += offset_y
    return np_contour.tolist()

def get_boundaries(tile_info: dict, args: argparse.Namespace, mask_shape: tuple):
    """
    Get slice boundaries of to-be-assigned mask to main mask.

    Args:
        tile_info (dict): Tile properties.
        args (argparse.Namespace): Parsed command-line arguments.
        mask_shape (tuple): Exact sizes of the mask.

    Returns:
        tuple: A tuple containing tile properties.
            - lower_y (int): The lower bound of the y-coordinate.
            - upper_y (int): The upper bound of the y-coordinate.
            - lower_x (int): The lower bound of the x-coordinate.
            - upper_x (int): The upper bound of the x-coordinate.
    """
    lower_x = round((tile_info["x"] - args.roi_x) / args.ds)
    diff_x = abs(lower_x) if lower_x < 0 else 0

    lower_x = 0 if lower_x < 0 else lower_x
    upper_x = lower_x + mask_shape[1] + diff_x

    lower_y = round((tile_info["y"] - args.roi_y) / args.ds)
    diff_y = abs(lower_y) if lower_y < 0 else 0

    lower_y = 0 if lower_y < 0 else lower_y
    upper_y = lower_y + mask_shape[0] + diff_y

    assert upper_x - lower_x == mask_shape[1]
    assert upper_y - lower_y == mask_shape[0]
    return (lower_y, upper_y, lower_x, upper_x)


def extract_tile_info(tile_name: str):
    """
    Extract tile information from tile file name.

    Args:
        tile_name (str): Name of the tile file.

    Returns:
        dict: Tile properties.
    """
    pattern = r"\[(.*?)\]"
    matches = re.findall(pattern, tile_name)
    if matches:
        matched_string = matches[0]
        split_string = matched_string.split(",")
        extracted_dict = {}
        extracted_dict["file_name"] = tile_name
        for item in split_string:
            key, value = item.split("=")
            extracted_dict[key] = float(value) if key == "ds" else int(value)
        return extracted_dict
    else:
        return None

def xy_to_geojson(polygons: list, output_path: str, properties: dict):
    """
    Save vector polygons to GeoJSON format.
    """
    features = []
    for coords in polygons:
        if len(coords) < 3:
            continue

        if not (coords[0] == coords[-1]).all():
            coords = np.vstack([coords, coords[0]])

        polygon = Polygon(coords)
        if not polygon.is_valid:
            continue

        features.append(geojson.Feature(geometry=polygon, properties=properties))

    feature_collection = geojson.FeatureCollection(features)
    logging.info(f"Exporting {len(features)} vector objects to GeoJSON.")
    with open(output_path, "w") as f:
        f.write(geojson.dumps(feature_collection, indent=2))

def predict(config: dict, args: argparse.Namespace, operation: str, k_size: int):
    """
    Run inference and return eroded, offset polygons using pred.masks.xy.

    Args:
        config (dict): Paths and model config.
        args (argparse.Namespace): Parsed CLI args with ROI and scaling.
        operation (str): Unused.
        k_size (int): Buffer size used for erosion.

    Returns:
        None
    """
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
    )
    logging.info("Inference done. Extracting polygons...")

    polygons = []
    erosion_buffer = -k_size  # equivalent to your .buffer(-6)

    for pred in preds:
        tile_info = extract_tile_info(Path(pred.path).stem)
        if tile_info is None or pred.masks is None or pred.masks.xy is None:
            continue

        for xy in pred.masks.xy:
            if len(xy) < 3:
                continue

            # Scale + shift polygon to full image space
            coords = xy * args.ds
            coords[:,0] += tile_info["x"] #+ args.roi_x
            coords[:,1] += tile_info["y"] #+ args.roi_y
            polygon = Polygon(coords).buffer(erosion_buffer)

            if polygon.is_empty:
                continue

            # Support MultiPolygons by splitting them up
            if polygon.geom_type == 'MultiPolygon':
                for geom in polygon.geoms:
                    if geom.is_valid and not geom.is_empty:
                        polygons.append(np.array(geom.exterior.coords))
            elif polygon.geom_type == 'Polygon':
                if polygon.is_valid:
                    polygons.append(np.array(polygon.exterior.coords))

    logging.info(f"Exporting {len(polygons)} eroded polygons to GeoJSON.")
    xy_to_geojson(polygons, config["preds_path"], config["geojson_properties"])

def load_arguments():
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
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
    args = parser.parse_args()
    return args
    
def main():
    if not check_version():
        logging.error("PYTHON VERSION SHOULD BE AT LEAST 3.8!")
        return

    logging.info("Process started.")
    start_time = time.time()

    args = load_arguments()
    config_path = args.base_path / "config.json"
    config = read_config(config_path, args.base_path, logging.getLogger(__name__))

    predict(
        config,
        args,
        config["morphology"]["operation"],
        config["morphology"]["kernel_size"],
    )

    logging.info(f"Process finished in {time.time() - start_time:.2f} seconds.")


if __name__ == "__main__":
    main()