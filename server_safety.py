"""Bounded input/output and stable public failures. No model imports."""
import base64
import io
import json
import math
import warnings
from PIL import Image, UnidentifiedImageError

ERRORS = {
    "UPLOAD_TOO_LARGE": (413, "이미지 파일이 너무 큽니다."),
    "REQUEST_TOO_LARGE": (413, "요청 크기가 허용 범위를 넘었습니다."),
    "IMAGE_TOO_LARGE": (413, "이미지 해상도가 허용 범위를 넘었습니다."),
    "INVALID_IMAGE": (200, "이미지를 읽지 못했습니다. PNG, JPG 또는 WebP 파일을 확인해 주세요."),
    "INVALID_PROMPT": (200, "선택 좌표를 확인한 뒤 다시 시도해 주세요."),
    "INVALID_REQUEST": (422, "요청 형식을 확인해 주세요."),
    "RESPONSE_TOO_LARGE": (413, "처리 결과가 너무 큽니다. 더 작은 이미지로 다시 시도해 주세요."),
    "MODEL_UNAVAILABLE": (503, "이미지 처리 서비스를 준비하지 못했습니다. 잠시 후 다시 시도해 주세요."),
    "INFERENCE_FAILED": (200, "이미지를 처리하지 못했습니다. 잠시 후 다시 시도해 주세요."),
    "INTERNAL_ERROR": (500, "요청을 처리하지 못했습니다. 잠시 후 다시 시도해 주세요."),
}


class PublicError(Exception):
    def __init__(self, code):
        self.code = code if code in ERRORS else "INTERNAL_ERROR"
        self.status, self.message = ERRORS[self.code]
        super().__init__(self.code)

    def payload(self):
        return {"ok": False, "code": self.code, "error": self.message}


def read_image(file, settings):
    raw = file.read(settings.max_upload_bytes + 1)
    if len(raw) > settings.max_upload_bytes:
        raise PublicError("UPLOAD_TOO_LARGE")
    signature = ("PNG" if raw.startswith(b"\x89PNG\r\n\x1a\n") else
                 "JPEG" if raw.startswith(b"\xff\xd8\xff") else
                 "WEBP" if raw[:4] == b"RIFF" and raw[8:12] == b"WEBP" else None)
    if signature is None:
        raise PublicError("INVALID_IMAGE")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(raw)) as image:
                if image.format != signature or getattr(image, "is_animated", False):
                    raise PublicError("INVALID_IMAGE")
                width, height = image.size
                if width > settings.max_width or height > settings.max_height or width * height > settings.max_pixels:
                    raise PublicError("IMAGE_TOO_LARGE")
                image.load()
                return image.convert("RGBA"), len(raw)
    except (Image.DecompressionBombError, Image.DecompressionBombWarning):
        raise PublicError("IMAGE_TOO_LARGE") from None
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError):
        raise PublicError("INVALID_IMAGE") from None


def coordinate(value, limit):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PublicError("INVALID_PROMPT")
    try:
        valid = math.isfinite(value) and 0 <= value <= limit - 1
    except (OverflowError, ValueError):
        valid = False
    if not valid:
        raise PublicError("INVALID_PROMPT")
    return int(round(value))


def prompt_points(raw, width, height, settings):
    if raw is None or raw == "":
        return []
    if len(raw.encode("utf-8")) > settings.max_prompt_bytes:
        raise PublicError("INVALID_PROMPT")
    try:
        items = json.loads(raw)
    except (ValueError, RecursionError):
        raise PublicError("INVALID_PROMPT") from None
    if not isinstance(items, list) or len(items) > settings.max_points:
        raise PublicError("INVALID_PROMPT")
    result = []
    for item in items:
        if not isinstance(item, dict) or set(item) != {"x", "y"}:
            raise PublicError("INVALID_PROMPT")
        result.append((coordinate(item["x"], width), coordinate(item["y"], height)))
    return result


def validate_prompt(width, height, settings, point_x=None, point_y=None, box_x1=None,
                    box_y1=None, box_x2=None, box_y2=None, positive_points_json=None, negative_points_json=None):
    positive = prompt_points(positive_points_json, width, height, settings)
    negative = prompt_points(negative_points_json, width, height, settings)
    point = None
    if point_x is not None or point_y is not None:
        point = (coordinate(point_x, width), coordinate(point_y, height))
    box = None
    if any(x is not None for x in (box_x1, box_y1, box_x2, box_y2)):
        x1, x2 = sorted((coordinate(box_x1, width), coordinate(box_x2, width)))
        y1, y2 = sorted((coordinate(box_y1, height), coordinate(box_y2, height)))
        if x1 == x2 or y1 == y2:
            raise PublicError("INVALID_PROMPT")
        box = (x1, y1, x2, y2)
    if not positive and not negative and point is None and box is None:
        raise PublicError("INVALID_PROMPT")
    return point, box, positive, negative


class BoundedPNG(io.BytesIO):
    def __init__(self, limit):
        super().__init__()
        self.limit = limit

    def write(self, data):
        if self.tell() + len(data) > self.limit:
            raise PublicError("RESPONSE_TOO_LARGE")
        return super().write(data)


class ResponseBudget:
    """Account for all PNGs before base64 allocation, reserving room for JSON."""
    def __init__(self, max_bytes):
        self.max_bytes = max_bytes
        self.encoded_bytes = 0
        self.binary_bytes = 0

    def png(self, image):
        remaining = self.max_bytes - 65536 - self.encoded_bytes
        with BoundedPNG(max(0, remaining // 4 * 3)) as buffer:
            image.save(buffer, format="PNG")
            binary = buffer.getvalue()
        encoded_size = 4 * ((len(binary) + 2) // 3)
        if encoded_size > remaining:
            raise PublicError("RESPONSE_TOO_LARGE")
        self.binary_bytes += len(binary)
        self.encoded_bytes += encoded_size
        return base64.b64encode(binary).decode("ascii")

    def render(self, payload):
        # Serialize once, then send these exact bytes. Never log payloads.
        result = json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")
        if len(result) > self.max_bytes:
            raise PublicError("RESPONSE_TOO_LARGE")
        return result
