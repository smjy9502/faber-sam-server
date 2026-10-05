import base64
import io
import json
from pathlib import Path
from typing import Optional, Any

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from alpha_preservation import preserve_alpha
from segment_anything import SamPredictor, sam_model_registry
from torchvision import transforms

try:
    from transformers import AutoModelForImageSegmentation
except ImportError:
    AutoModelForImageSegmentation = None


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "sam_vit_b_01ec64.pth"
MODEL_TYPE = "vit_b"
BIREFNET_MODEL_ID = "ZhengPeng7/BiRefNet"
BIREFNET_INPUT_SIZE = (1024, 1024)

app = FastAPI(title="FABER SAM Server", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"

    if torch.backends.mps.is_available():
        return "mps"

    return "cpu"


DEVICE = get_device()

sam = sam_model_registry[MODEL_TYPE](checkpoint=str(MODEL_PATH))
sam.to(device=DEVICE)
predictor = SamPredictor(sam)

_birefnet_model = None
_birefnet_device = None

BIREFNET_TRANSFORM = transforms.Compose(
    [
        transforms.Resize(BIREFNET_INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225],
        ),
    ]
)


def get_birefnet_model():
    global _birefnet_model, _birefnet_device

    if _birefnet_model is not None:
        return _birefnet_model, _birefnet_device

    if AutoModelForImageSegmentation is None:
        raise RuntimeError(
            "BiRefNet 의존성이 설치되지 않았습니다. requirements.txt를 다시 설치해주세요."
        )

    model = AutoModelForImageSegmentation.from_pretrained(
        BIREFNET_MODEL_ID,
        trust_remote_code=True,
        revision="e2bf8e4460fc8fa32bba5ea4d94b3233d367b0e4",
    )
    model.eval()

    device = DEVICE
    model.to(device=device)

    _birefnet_model = model
    _birefnet_device = device
    return _birefnet_model, _birefnet_device


@app.get("/health")
def health():
    return {
        "ok": True,
        "model_type": MODEL_TYPE,
        "device": DEVICE,
        "model_path_exists": MODEL_PATH.exists(),
        "birefnet_model_id": BIREFNET_MODEL_ID,
        "birefnet_loaded": _birefnet_model is not None,
    }


def read_image(uploaded_file: UploadFile) -> np.ndarray:
    image_bytes = uploaded_file.file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return np.array(image)


def encode_mask_png(mask: np.ndarray) -> str:
    mask_uint8 = (mask.astype(np.uint8) * 255)

    success, encoded = cv2.imencode(".png", mask_uint8)

    if not success:
        raise RuntimeError("마스크 PNG 인코딩에 실패했습니다.")

    return base64.b64encode(encoded.tobytes()).decode("utf-8")



def encode_gray_png(mask_uint8: np.ndarray) -> str:
    image = Image.fromarray(mask_uint8.astype(np.uint8), mode="L")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def encode_rgba_png(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def run_birefnet(image: Image.Image) -> tuple[np.ndarray, Image.Image]:
    model, device = get_birefnet_model()

    rgb_image = image.convert("RGB")
    original_size = rgb_image.size
    input_tensor = BIREFNET_TRANSFORM(rgb_image).unsqueeze(0).to(device)

    with torch.inference_mode():
        prediction = model(input_tensor)[-1].sigmoid().detach().cpu()[0].squeeze()

    prediction = prediction.clamp(0, 1)
    mask_small = (prediction.numpy() * 255.0).astype(np.uint8)
    mask_image = Image.fromarray(mask_small, mode="L").resize(
        original_size,
        Image.Resampling.LANCZOS,
    )
    mask_image, cutout = preserve_alpha(image, mask_image)
    mask_uint8 = np.array(mask_image, dtype=np.uint8)

    return mask_uint8, cutout


@app.post("/remove-background")
def remove_background(image: UploadFile = File(...)):
    image_bytes = image.file.read()

    try:
        source_image = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
    except Exception as error:
        return {
            "ok": False,
            "error": f"이미지를 읽지 못했습니다: {error}",
        }

    try:
        mask_uint8, cutout = run_birefnet(source_image)
    except Exception as error:
        return {
            "ok": False,
            "error": f"BiRefNet 추론에 실패했습니다: {error}",
        }

    binary_mask = mask_uint8 >= 128
    bbox = get_mask_bbox(binary_mask)

    return {
        "ok": True,
        "engine": "birefnet",
        "model_id": BIREFNET_MODEL_ID,
        "width": source_image.width,
        "height": source_image.height,
        "bbox": bbox,
        "mask_png_base64": encode_gray_png(mask_uint8),
        "cutout_png_base64": encode_rgba_png(cutout),
    }

def get_mask_bbox(mask: np.ndarray) -> Optional[dict]:
    ys, xs = np.where(mask)

    if len(xs) == 0 or len(ys) == 0:
        return None

    min_x = int(xs.min())
    min_y = int(ys.min())
    max_x = int(xs.max())
    max_y = int(ys.max())

    return {
        "x": min_x,
        "y": min_y,
        "width": max_x - min_x + 1,
        "height": max_y - min_y + 1,
    }


def get_mask_metrics(mask: np.ndarray, image_width: int, image_height: int) -> dict:
    bbox = get_mask_bbox(mask)
    area = int(mask.sum())
    image_area = image_width * image_height

    if bbox is None or area == 0:
        return {
            "area": 0,
            "area_ratio": 0,
            "bbox": None,
            "bbox_area_ratio": 0,
            "fill_ratio": 0,
            "edge_touch_count": 0,
            "background_like_penalty": 999,
        }

    bbox_area = bbox["width"] * bbox["height"]
    edge_touch_count = 0

    if bbox["x"] <= 2:
        edge_touch_count += 1
    if bbox["y"] <= 2:
        edge_touch_count += 1
    if bbox["x"] + bbox["width"] >= image_width - 2:
        edge_touch_count += 1
    if bbox["y"] + bbox["height"] >= image_height - 2:
        edge_touch_count += 1

    area_ratio = area / image_area
    bbox_area_ratio = bbox_area / image_area
    fill_ratio = area / bbox_area if bbox_area else 0

    background_like_penalty = 0.0

    if area_ratio > 0.72:
        background_like_penalty += 4.0
    if bbox_area_ratio > 0.82 and fill_ratio > 0.55:
        background_like_penalty += 4.0
    if edge_touch_count >= 3 and area_ratio > 0.35:
        background_like_penalty += 3.0
    if fill_ratio > 0.88 and bbox_area_ratio > 0.45:
        background_like_penalty += 2.0

    return {
        "area": area,
        "area_ratio": area_ratio,
        "bbox": bbox,
        "bbox_area_ratio": bbox_area_ratio,
        "fill_ratio": fill_ratio,
        "edge_touch_count": edge_touch_count,
        "background_like_penalty": background_like_penalty,
    }


def choose_best_mask(
    masks: np.ndarray,
    scores: np.ndarray,
    image_width: int,
    image_height: int,
    prompt_box: Optional[tuple[int, int, int, int]] = None,
) -> tuple[int, np.ndarray, float, list[dict]]:
    candidates = []

    prompt_area = None
    if prompt_box is not None:
        x1, y1, x2, y2 = prompt_box
        prompt_area = max(1, (x2 - x1) * (y2 - y1))

    for index, mask in enumerate(masks):
        metrics = get_mask_metrics(mask, image_width, image_height)
        sam_score = float(scores[index])

        selection_score = sam_score
        selection_score -= metrics["background_like_penalty"]

        # Prefer masks that are neither tiny specks nor image-filling backgrounds.
        if 0.005 <= metrics["area_ratio"] <= 0.65:
            selection_score += 0.3

        # For box prompts, prefer a candidate whose bbox is reasonably related to the prompt box.
        if prompt_area and metrics["bbox"] is not None:
            mask_box_area = metrics["bbox"]["width"] * metrics["bbox"]["height"]
            box_ratio = mask_box_area / prompt_area

            if 0.15 <= box_ratio <= 1.25:
                selection_score += 0.25
            elif box_ratio > 1.6:
                selection_score -= 0.8

        candidates.append(
            {
                "index": index,
                "sam_score": sam_score,
                "selection_score": float(selection_score),
                "metrics": metrics,
            }
        )

    candidates.sort(key=lambda candidate: candidate["selection_score"], reverse=True)

    best_index = int(candidates[0]["index"])
    best_mask = masks[best_index]
    best_score = float(scores[best_index])

    return best_index, best_mask, best_score, candidates



def parse_prompt_points(points_json: Optional[str], width: int, height: int) -> list[tuple[int, int]]:
    if not points_json:
        return []

    try:
        raw_points: Any = json.loads(points_json)
    except json.JSONDecodeError:
        return []

    if not isinstance(raw_points, list):
        return []

    points: list[tuple[int, int]] = []

    for item in raw_points:
        if not isinstance(item, dict):
            continue

        try:
            x = int(round(float(item.get("x"))))
            y = int(round(float(item.get("y"))))
        except (TypeError, ValueError):
            continue

        x = max(0, min(width - 1, x))
        y = max(0, min(height - 1, y))
        points.append((x, y))

    return points


@app.post("/segment")
def segment(
    image: UploadFile = File(...),
    point_x: Optional[float] = Form(None),
    point_y: Optional[float] = Form(None),
    box_x1: Optional[float] = Form(None),
    box_y1: Optional[float] = Form(None),
    box_x2: Optional[float] = Form(None),
    box_y2: Optional[float] = Form(None),
    positive_points_json: Optional[str] = Form(None),
    negative_points_json: Optional[str] = Form(None),
):
    """
    Supports two SAM prompt modes.

    1) Box prompt, recommended for product cutout:
       box_x1, box_y1, box_x2, box_y2 in original image pixel coordinates.
       Use this when the user roughly drags around the full object.

    2) Point prompt, fallback:
       point_x, point_y in original image pixel coordinates.

    3) Refinement prompt:
       positive_points_json and negative_points_json as JSON arrays:
       [{"x": 120, "y": 80}, ...]
    """

    np_image = read_image(image)
    height, width = np_image.shape[:2]

    predictor.set_image(np_image)

    positive_points = parse_prompt_points(positive_points_json, width, height)
    negative_points = parse_prompt_points(negative_points_json, width, height)
    has_refinement_points = len(positive_points) + len(negative_points) > 0

    point_coords = None
    point_labels = None

    if has_refinement_points:
        all_points = positive_points + negative_points
        labels = [1] * len(positive_points) + [0] * len(negative_points)
        point_coords = np.array(all_points, dtype=np.float32)
        point_labels = np.array(labels, dtype=np.int32)

    has_box = all(
        value is not None
        for value in [box_x1, box_y1, box_x2, box_y2]
    )

    prompt_box = None

    if has_box:
        x1 = max(0, min(width - 1, int(round(min(box_x1, box_x2)))))
        y1 = max(0, min(height - 1, int(round(min(box_y1, box_y2)))))
        x2 = max(0, min(width - 1, int(round(max(box_x1, box_x2)))))
        y2 = max(0, min(height - 1, int(round(max(box_y1, box_y2)))))

        if x2 <= x1 or y2 <= y1:
            return {
                "ok": False,
                "error": "유효하지 않은 박스 좌표입니다.",
                "width": width,
                "height": height,
            }

        input_box = np.array([x1, y1, x2, y2])
        prompt_box = (x1, y1, x2, y2)

        masks, scores, logits = predictor.predict(
            box=input_box,
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=True,
        )

        prompt = {
            "type": "refine" if has_refinement_points else "box",
            "box": {
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
            },
            "positive_points": [{"x": x, "y": y} for x, y in positive_points],
            "negative_points": [{"x": x, "y": y} for x, y in negative_points],
        }
    else:
        if has_refinement_points:
            masks, scores, logits = predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=True,
            )

            prompt = {
                "type": "refine",
                "positive_points": [{"x": x, "y": y} for x, y in positive_points],
                "negative_points": [{"x": x, "y": y} for x, y in negative_points],
            }
        else:
            if point_x is None or point_y is None:
                return {
                    "ok": False,
                    "error": "point, box 또는 보정 point 좌표가 필요합니다.",
                    "width": width,
                    "height": height,
                }

            x = max(0, min(width - 1, int(round(point_x))))
            y = max(0, min(height - 1, int(round(point_y))))

            input_point = np.array([[x, y]])
            input_label = np.array([1])

            masks, scores, logits = predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=True,
            )

            prompt = {
                "type": "point",
                "point": {
                    "x": x,
                    "y": y,
                },
            }

    best_index, best_mask, best_score, candidates = choose_best_mask(
        masks=masks,
        scores=scores,
        image_width=width,
        image_height=height,
        prompt_box=prompt_box,
    )

    mask_png_base64 = encode_mask_png(best_mask)
    bbox = get_mask_bbox(best_mask)

    return {
        "ok": True,
        "width": width,
        "height": height,
        "prompt": prompt,
        "score": best_score,
        "bbox": bbox,
        "selected_candidate_index": best_index,
        "candidates": candidates,
        "mask_png_base64": mask_png_base64,
    }