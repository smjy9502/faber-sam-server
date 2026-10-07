"""FABER inference API. Startup loads SAM; imports and tests never load weights."""
import io
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import UploadFile
from starlette.exceptions import HTTPException
from PIL import Image

from inference_runtime import InferenceRuntime, BIREFNET_MODEL_ID
from runtime_config import Settings
from server_safety import PublicError, ResponseBudget, read_image, validate_prompt

logger = logging.getLogger("faber.inference")


def error_response(error):
    return JSONResponse(error.payload(), status_code=error.status)


class RequestBoundary:
    """Bound the complete body before multipart parsing; log only an explicit allowlist."""
    def __init__(self, app, settings, runtime):
        self.app, self.settings, self.runtime = app, settings, runtime

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        request_id = uuid.uuid4().hex
        started = time.monotonic()
        state = scope.setdefault("state", {})
        status, response_started = 500, False
        path = scope.get("path", "")
        endpoint = path if path in {"/health", "/ready", "/segment", "/remove-background"} else "other"

        async def tracked_send(message):
            nonlocal status, response_started
            if message["type"] == "http.response.start":
                status, response_started = message["status"], True
                message["headers"] = list(message.get("headers", [])) + [
                    (b"x-request-id", request_id.encode()), (b"cache-control", b"no-store")]
            await send(message)

        try:
            if scope.get("method") == "POST" and endpoint in {"/segment", "/remove-background"}:
                lengths = [v for k, v in scope.get("headers", []) if k.lower() == b"content-length"]
                if lengths:
                    if len(lengths) != 1 or not lengths[0].isdigit():
                        raise PublicError("INVALID_REQUEST")
                    if int(lengths[0]) > self.settings.max_request_bytes:
                        raise PublicError("REQUEST_TOO_LARGE")
                body = io.BytesIO()
                while True:
                    message = await receive()
                    if message["type"] == "http.disconnect":
                        return
                    chunk = message.get("body", b"")
                    if body.tell() + len(chunk) > self.settings.max_request_bytes:
                        raise PublicError("REQUEST_TOO_LARGE")
                    body.write(chunk)
                    if not message.get("more_body", False):
                        break
                state["request_bytes"] = body.tell()
                # Release the buffer immediately after the parser receives it.
                async def bounded_receive():
                    nonlocal body
                    if body is not None:
                        data = body.getvalue()
                        body.close()
                        body = None
                        return {"type": "http.request", "body": data, "more_body": False}
                    return await receive()
                await self.app(scope, bounded_receive, tracked_send)
            else:
                await self.app(scope, receive, tracked_send)
        except PublicError as error:
            state["error_code"] = error.code
            if not response_started:
                await error_response(error)(scope, receive, tracked_send)
        except Exception:
            state["error_code"] = "INTERNAL_ERROR"
            if not response_started:
                await error_response(PublicError("INTERNAL_ERROR"))(scope, receive, tracked_send)
        finally:
            record = {"request_id": request_id, "endpoint": endpoint, "status": status,
                      "latency_ms": round((time.monotonic() - started) * 1000),
                      "sam_ready": self.runtime.sam_ready, "birefnet_loaded": self.runtime.birefnet_loaded}
            for key in ("request_bytes", "image_width", "image_height", "image_bytes", "error_code"):
                if key in state:
                    record[key] = state[key]
            # Do not emit routine successful health probes.
            if endpoint not in {"/health", "/ready"} or status >= 400:
                log = logger.warning if status >= 400 or "error_code" in state else logger.info
                log(json.dumps(record, separators=(",", ":")))


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



def segment_result(source, fields, runtime, settings, budget):
    width, height = source.size
    parsed = {}
    for key in ("point_x", "point_y", "box_x1", "box_y1", "box_x2", "box_y2"):
        value = fields.get(key)
        try:
            parsed[key] = None if value is None else float(value)
        except (TypeError, ValueError):
            raise PublicError("INVALID_PROMPT") from None
    point, box, positive, negative = validate_prompt(
        width, height, settings, **parsed,
        positive_points_json=fields.get("positive_points_json"),
        negative_points_json=fields.get("negative_points_json"))
    refined = bool(positive or negative)
    coordinates = np.array(positive + negative, dtype=np.float32) if refined else None
    labels = np.array([1] * len(positive) + [0] * len(negative), dtype=np.int32) if refined else None
    if box is not None or refined:
        prompt = {"type": "refine" if refined else "box",
                  "positive_points": [{"x": x, "y": y} for x, y in positive],
                  "negative_points": [{"x": x, "y": y} for x, y in negative]}
        if box:
            prompt["box"] = dict(zip(("x1", "y1", "x2", "y2"), box))
    else:
        coordinates, labels = np.array([point]), np.array([1])
        prompt = {"type": "point", "point": {"x": point[0], "y": point[1]}}
    masks, scores, _ = runtime.segment(np.array(source.convert("RGB")),
        box=np.array(box) if box else None, point_coords=coordinates,
        point_labels=labels, multimask_output=True)
    index, mask, score, candidates = choose_best_mask(masks, scores, width, height, box)
    return {"ok": True, "width": width, "height": height, "prompt": prompt,
            "score": score, "bbox": get_mask_bbox(mask), "selected_candidate_index": index,
            "candidates": candidates, "mask_png_base64": budget.png(Image.fromarray(mask.astype(np.uint8) * 255))}


def process_image(file, fields, endpoint, runtime, settings, state):
    source, size = read_image(file, settings)
    state.update(image_width=source.width, image_height=source.height, image_bytes=size)
    budget = ResponseBudget(settings.max_response_bytes)
    try:
        if endpoint == "segment":
            result = segment_result(source, fields, runtime, settings, budget)
        else:
            mask, cutout = runtime.remove_background(source)
            result = {"ok": True, "engine": "birefnet", "model_id": BIREFNET_MODEL_ID,
                      "width": source.width, "height": source.height,
                      "bbox": get_mask_bbox(mask >= 128),
                      "mask_png_base64": budget.png(Image.fromarray(mask.astype(np.uint8))),
                      "cutout_png_base64": budget.png(cutout)}
        return Response(budget.render(result), media_type="application/json")
    except PublicError:
        raise
    except Exception:
        raise PublicError("INFERENCE_FAILED") from None
    finally:
        source.close()


def create_app(settings=None, runtime=None):
    settings = settings or Settings.from_env()
    runtime = runtime or InferenceRuntime(settings)

    @asynccontextmanager
    async def lifespan(app):
        try:
            await run_in_threadpool(runtime.start)
        except Exception as error:
            code = "SAM_CHECKPOINT_MISSING" if isinstance(error, RuntimeError) and str(error) == "SAM_CHECKPOINT_MISSING" else "MODEL_STARTUP_FAILED"
            logger.error(json.dumps({"event": "startup_failed", "code": code}))
            raise RuntimeError(code) from None
        yield

    app = FastAPI(title="FABER SAM Server", version="0.3.0", lifespan=lifespan)
    app.state.runtime = runtime

    @app.exception_handler(PublicError)
    async def public_error(request, error):
        request.scope.setdefault("state", {})["error_code"] = error.code
        return error_response(error)

    @app.exception_handler(HTTPException)
    async def http_error(request, error):
        if error.status_code in (404, 405):
            return JSONResponse({"ok": False, "code": "NOT_FOUND", "error": "지원하지 않는 요청입니다."}, status_code=error.status_code)
        return await public_error(request, PublicError("INVALID_REQUEST"))

    @app.get("/health")
    def health():
        return {"ok": True, "model_type": "vit_b", "device": runtime.device,
                "model_path_exists": settings.sam_checkpoint.is_file(),
                "sam_ready": runtime.sam_ready, "birefnet_model_id": BIREFNET_MODEL_ID,
                "birefnet_loaded": runtime.birefnet_loaded}

    @app.get("/ready")
    def ready():
        return JSONResponse({"ok": runtime.sam_ready, "sam_ready": runtime.sam_ready,
                             "birefnet_loaded": runtime.birefnet_loaded,
                             "birefnet_strategy": "lazy_offline"},
                            status_code=200 if runtime.sam_ready else 503)

    async def dispatch(request, endpoint):
        if not request.headers.get("content-type", "").lower().startswith("multipart/form-data;"):
            raise PublicError("INVALID_REQUEST")
        allowed = {"image"}
        if endpoint == "segment":
            allowed.update({"point_x", "point_y", "box_x1", "box_y1", "box_x2", "box_y2", "positive_points_json", "negative_points_json"})
        try:
            async with request.form(max_files=1, max_fields=8, max_part_size=settings.max_prompt_bytes) as form:
                if any(key not in allowed or len(form.getlist(key)) != 1 for key in form):
                    raise PublicError("INVALID_REQUEST")
                image = form.get("image")
                if not isinstance(image, UploadFile) or any(isinstance(v, UploadFile) for k, v in form.items() if k != "image"):
                    raise PublicError("INVALID_REQUEST")
                return await run_in_threadpool(process_image, image.file,
                    {k: v for k, v in form.items() if k != "image"}, endpoint, runtime, settings,
                    request.scope.setdefault("state", {}))
        except PublicError:
            raise
        except HTTPException:
            raise PublicError("INVALID_REQUEST") from None
        except Exception:
            raise PublicError("INVALID_REQUEST") from None

    @app.post("/segment")
    async def segment(request: Request):
        return await dispatch(request, "segment")

    @app.post("/remove-background")
    async def remove_background(request: Request):
        return await dispatch(request, "remove-background")

    app.add_middleware(RequestBoundary, settings=settings, runtime=runtime)
    app.add_middleware(CORSMiddleware, allow_origins=list(settings.cors_origins),
                       allow_credentials=True, allow_methods=["GET", "POST"], allow_headers=["Content-Type"])
    return app


app = create_app()
