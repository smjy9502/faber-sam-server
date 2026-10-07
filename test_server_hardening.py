"""Offline tests: synthetic images, fake predictors/loaders, actual ASGI routing.
Run with an existing environment; never import test_segment (it invokes a server).
"""
import asyncio
import base64
import io
import json
import logging
from pathlib import Path
import sys
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from unittest.mock import patch

import numpy as np
from PIL import Image
from inference_runtime import InferenceRuntime, LazyResource
from runtime_config import Settings, BIREFNET_REVISION
from server_safety import PublicError, ResponseBudget, read_image, validate_prompt
from main import create_app


def png(size=(8, 8), mode="RGBA"):
    out = io.BytesIO()
    Image.new(mode, size, (10, 20, 30, 128) if mode == "RGBA" else (10, 20, 30)).save(out, format="PNG")
    return out.getvalue()


class FakeRuntime:
    device = "cpu"
    sam_ready = True
    birefnet_loaded = False

    def __init__(self):
        self.calls = []
        self.failure = False

    def start(self):
        self.sam_ready = True

    def segment(self, image, **kwargs):
        self.calls.append(kwargs)
        if self.failure:
            raise RuntimeError("SECRET /private/model/cache token=private")
        mask = np.zeros(image.shape[:2], dtype=bool)
        mask[2:6, 2:6] = True
        return np.array([mask]), np.array([0.9]), None

    def remove_background(self, source):
        self.calls.append("remove")
        if self.failure:
            raise RuntimeError("SECRET /private/model/cache token=private")
        from alpha_preservation import preserve_alpha
        mask, result = preserve_alpha(source, Image.new("L", source.size, 200))
        return np.array(mask), result


def multipart(fields=None, image=None):
    boundary = "faber-test-boundary"
    data = bytearray()
    for key, value in (fields or []):
        data.extend(f'--{boundary}\r\nContent-Disposition: form-data; name="{key}"\r\n\r\n{value}\r\n'.encode())
    if image is not None:
        data.extend(f'--{boundary}\r\nContent-Disposition: form-data; name="image"; filename="private-name.png"\r\nContent-Type: image/png\r\n\r\n'.encode())
        data.extend(image)
        data.extend(b"\r\n")
    data.extend(f'--{boundary}--\r\n'.encode())
    return bytes(data), f"multipart/form-data; boundary={boundary}".encode()


async def asgi_request(app, path, method="GET", body=b"", content_type=None, extra_headers=(), chunk_size=None):
    headers = list(extra_headers)
    if content_type:
        headers.append((b"content-type", content_type))
    scope = {"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1",
             "method": method, "scheme": "http", "path": path, "raw_path": path.encode(),
             "query_string": b"", "headers": headers, "server": ("test", 80), "client": ("test", 1)}
    chunks = [body[i:i + chunk_size] for i in range(0, len(body), chunk_size)] if chunk_size else [body]
    messages = []
    async def receive():
        if chunks:
            part = chunks.pop(0)
            return {"type": "http.request", "body": part, "more_body": bool(chunks)}
        return {"type": "http.disconnect"}
    async def send(message):
        messages.append(message)
    await app(scope, receive, send)
    start = next(m for m in messages if m["type"] == "http.response.start")
    raw = b"".join(m.get("body", b"") for m in messages if m["type"] == "http.response.body")
    return start["status"], json.loads(raw) if raw else None, dict(start["headers"])


class ConfigTest(unittest.TestCase):
    def test_defaults(self):
        self.assertEqual(Settings.from_env({}).port, 8080)
        self.assertEqual(Settings().birefnet_revision, BIREFNET_REVISION)

    def test_env_override(self):
        s = Settings.from_env({"PORT": "9000", "TORCH_THREADS": "1", "CORS_ORIGINS": "", "SAM_CHECKPOINT_PATH": "/models/test.pth"})
        self.assertEqual((s.port, s.torch_threads, s.cors_origins), (9000, 1, ()))
        self.assertEqual(s.sam_checkpoint, Path("/models/test.pth"))

    def test_invalid_or_excessive_config(self):
        for env in ({"PORT": "bad"}, {"MAX_PIXELS": "0"}, {"MAX_UPLOAD_BYTES": str(32 * 1024**2)},
                    {"MAX_RESPONSE_BYTES": str(32 * 1024**2)}, {"CORS_ORIGINS": "*"},
                    {"BIREFNET_REVISION": "main"}, {"LOG_LEVEL": "DEBUG"}, {"TORCH_THREADS": "99"}):
            with self.subTest(env=env), self.assertRaises(ValueError): Settings.from_env(env)

    def test_request_budget_must_exceed_file_budget(self):
        with self.assertRaises(ValueError): Settings(max_request_bytes=10)


class ValidationTest(unittest.TestCase):
    def test_valid_alpha_decode(self):
        image, count = read_image(io.BytesIO(png()), Settings())
        self.assertEqual(image.getpixel((0, 0)), (10, 20, 30, 128))
        self.assertGreater(count, 0)

    def test_oversized_bytes(self):
        with self.assertRaisesRegex(PublicError, "UPLOAD_TOO_LARGE"):
            read_image(io.BytesIO(png()), Settings(max_upload_bytes=20))

    def test_oversized_pixels_before_decode(self):
        with self.assertRaisesRegex(PublicError, "IMAGE_TOO_LARGE"):
            read_image(io.BytesIO(png()), Settings(max_pixels=63))

    def test_width_and_height_limits(self):
        for settings in [Settings(max_width=7), Settings(max_height=7)]:
            with self.assertRaisesRegex(PublicError, "IMAGE_TOO_LARGE"): read_image(io.BytesIO(png()), settings)

    def test_invalid_signature_and_corrupt_image(self):
        for raw in [b"not an image", b"\x89PNG\r\n\x1a\ninvalid", b"GIF89a"]:
            with self.assertRaisesRegex(PublicError, "INVALID_IMAGE"): read_image(io.BytesIO(raw), Settings())

    def test_jpeg_and_webp(self):
        for fmt in ["JPEG", "WEBP"]:
            b = io.BytesIO(); Image.new("RGB", (8, 8)).save(b, format=fmt)
            image, _ = read_image(io.BytesIO(b.getvalue()), Settings())
            self.assertEqual(image.size, (8, 8))

    def test_valid_point_box_refinement(self):
        p = validate_prompt(8, 8, Settings(), box_x1=7, box_y1=7, box_x2=0, box_y2=0,
                            positive_points_json='[{"x":2,"y":3}]', negative_points_json='[{"x":0,"y":0}]')
        self.assertEqual(p[1:], ((0, 0, 7, 7), [(2, 3)], [(0, 0)]))

    def test_excessive_positive_and_negative(self):
        for name in ["positive_points_json", "negative_points_json"]:
            with self.assertRaisesRegex(PublicError, "INVALID_PROMPT"):
                validate_prompt(8, 8, Settings(), **{name: json.dumps([{"x": 1, "y": 1}] * 65)})

    def test_nonfinite_and_range(self):
        for value in [float("nan"), float("inf"), float("-inf"), 10**400, -1, 8, True, "1"]:
            with self.subTest(value=value), self.assertRaisesRegex(PublicError, "INVALID_PROMPT"):
                validate_prompt(8, 8, Settings(), point_x=value, point_y=1)

    def test_nonfinite_json(self):
        for raw in ['[{"x":NaN,"y":1}]', '[{"x":Infinity,"y":1}]']:
            with self.assertRaises(PublicError): validate_prompt(8, 8, Settings(), positive_points_json=raw)

    def test_invalid_box(self):
        for kwargs in [dict(box_x1=1), dict(box_x1=1, box_y1=1, box_x2=1, box_y2=3),
                       dict(box_x1=1, box_y1=1, box_x2=9, box_y2=3)]:
            with self.assertRaises(PublicError): validate_prompt(8, 8, Settings(), **kwargs)

    def test_malformed_prompt(self):
        for raw in ['{}', '[1]', '[{"x":1}]', 'bad', '[{"x":1,"y":1,"secret":1}]', ' ' * 16385]:
            with self.assertRaises(PublicError): validate_prompt(8, 8, Settings(), positive_points_json=raw)

    def test_missing_prompt(self):
        with self.assertRaises(PublicError): validate_prompt(8, 8, Settings())


class RuntimeTest(unittest.TestCase):
    def test_predictor_critical_section(self):
        events = []
        class Predictor:
            def set_image(self, image):
                self.image = image; events.append(("set", image)); time.sleep(0.002)
            def predict(self, **kwargs):
                image = self.image; time.sleep(0.002); events.append(("predict", image)); return image
        runtime = InferenceRuntime(Settings(), sam_loader=Predictor)
        runtime.start()
        with ThreadPoolExecutor(max_workers=8) as pool:
            self.assertEqual(list(pool.map(runtime.segment, range(16))), list(range(16)))
        for i in range(0, len(events), 2):
            self.assertEqual(events[i][0], "set")
            self.assertEqual(events[i + 1], ("predict", events[i][1]))

    def test_lazy_loader_once(self):
        calls = []
        def load(): calls.append(1); time.sleep(0.01); return object()
        resource = LazyResource(load)
        with ThreadPoolExecutor(max_workers=8) as pool: values = list(pool.map(lambda _: resource.get(), range(20)))
        self.assertEqual(len(calls), 1)
        self.assertTrue(all(v is values[0] for v in values))

    def test_lock_released_on_predict_failure(self):
        class Predictor:
            def set_image(self, image): pass
            def predict(self, **kwargs): raise ValueError("private")
        r = InferenceRuntime(Settings(), sam_loader=Predictor); r.start()
        for _ in range(2):
            with self.assertRaises(ValueError): r.segment(1)

    def test_missing_checkpoint_before_model_import(self):
        r = InferenceRuntime(Settings(sam_checkpoint=Path('/missing/faber-test.pth')))
        with self.assertRaisesRegex(RuntimeError, '^SAM_CHECKPOINT_MISSING$'): r.start()
        self.assertNotIn('torch', sys.modules)

    def test_model_load_has_offline_and_code_pin(self):
        # Fake transformers module exercises actual loader without importing a real model library.
        import types
        calls = []
        class Model:
            def eval(self): return self
            def to(self, **kwargs): return self
        def load(*args, **kwargs): calls.append(kwargs); return Model()
        fake = types.SimpleNamespace(AutoModelForImageSegmentation=types.SimpleNamespace(from_pretrained=load))
        with patch.dict(sys.modules, {'transformers': fake}):
            InferenceRuntime(Settings())._load_birefnet()
        self.assertTrue(calls[0]['local_files_only'])
        self.assertEqual(calls[0]['revision'], BIREFNET_REVISION)
        self.assertEqual(calls[0]['code_revision'], BIREFNET_REVISION)


class ResponseTest(unittest.TestCase):
    def test_png_and_json_budget(self):
        b = ResponseBudget(100_000)
        value = b.png(Image.new('L', (8, 8)))
        self.assertEqual(b.encoded_bytes, len(value))
        self.assertEqual(b.binary_bytes, len(base64.b64decode(value)))
        self.assertIn(b'"ok":true', b.render({'ok': True, 'mask_png_base64': value}))

    def test_png_oversize_before_base64(self):
        with patch('server_safety.base64.b64encode') as encoder:
            with self.assertRaisesRegex(PublicError, 'RESPONSE_TOO_LARGE'):
                ResponseBudget(65537).png(Image.new('L', (8, 8)))
            encoder.assert_not_called()

    def test_aggregate_png_budget(self):
        b = ResponseBudget(65700)
        b.png(Image.new('L', (8, 8)))
        with self.assertRaises(PublicError): b.png(Image.new('RGBA', (8, 8)))

    def test_serialized_response_size(self):
        with self.assertRaisesRegex(PublicError, 'RESPONSE_TOO_LARGE'):
            ResponseBudget(65537).render({'mask_png_base64': 'a' * 65537})


class ApiTest(unittest.TestCase):
    def setUp(self):
        self.runtime = FakeRuntime()
        self.app = create_app(Settings(), self.runtime)

    def post(self, path, fields=(), image=None, app=None, **kwargs):
        raw, content_type = multipart(list(fields), png() if image is None else image)
        return asyncio.run(asgi_request(app or self.app, path, 'POST', raw, content_type, **kwargs))

    def test_valid_segment_contract(self):
        status, d, headers = self.post('/segment', [('point_x', '3'), ('point_y', '3')])
        self.assertEqual(status, 200); self.assertTrue(d['ok'])
        self.assertTrue({'width','height','prompt','score','bbox','selected_candidate_index','candidates','mask_png_base64'} <= d.keys())
        mask = Image.open(io.BytesIO(base64.b64decode(d['mask_png_base64'])))
        self.assertEqual(mask.getpixel((3, 3)), 255)
        self.assertIn(b'x-request-id', headers)

    def test_box_and_refinement_contract(self):
        for fields in [[('box_x1','0'),('box_y1','0'),('box_x2','7'),('box_y2','7')],
                       [('positive_points_json','[{"x":3,"y":3}]'),('negative_points_json','[{"x":0,"y":0}]')]]:
            status, d, _ = self.post('/segment', fields)
            self.assertEqual(status, 200); self.assertTrue(d['ok'])

    def test_valid_background_contract_and_alpha(self):
        status, d, _ = self.post('/remove-background')
        self.assertEqual(status, 200); self.assertTrue(d['ok'])
        self.assertEqual(d['engine'], 'birefnet')
        image = Image.open(io.BytesIO(base64.b64decode(d['cutout_png_base64'])))
        self.assertEqual(image.getpixel((0,0)), (10,20,30,128))
        self.assertIn('mask_png_base64', d)

    def test_invalid_prompt_never_calls_model(self):
        for fields in [[('point_x','nan'),('point_y','1')], [('point_x','99'),('point_y','1')]]:
            status, d, _ = self.post('/segment', fields)
            self.assertEqual(status, 200); self.assertEqual(d['code'], 'INVALID_PROMPT')
        self.assertEqual(self.runtime.calls, [])

    def test_invalid_image_keeps_ok_false_contract(self):
        status, d, _ = self.post('/remove-background', image=b'private invalid data')
        self.assertEqual(status, 200); self.assertEqual(d['code'], 'INVALID_IMAGE')
        self.assertEqual(self.runtime.calls, [])

    def test_duplicate_box_field_rejected(self):
        status, d, _ = self.post('/segment', [('box_x1','0'),('box_x1','1')])
        self.assertEqual(status, 422); self.assertFalse(d['ok'])

    def test_missing_file_sanitized(self):
        raw, content_type = multipart()
        status, d, _ = asyncio.run(asgi_request(self.app, '/segment', 'POST', raw, content_type))
        self.assertEqual(status, 422); self.assertEqual(d['code'], 'INVALID_REQUEST')

    def test_request_size_header_and_stream(self):
        app = create_app(Settings(max_upload_bytes=100, max_request_bytes=200), self.runtime)
        for kwargs in [{'extra_headers':[(b'content-length',b'999')]}, {'chunk_size': 31}]:
            status, d, _ = self.post('/remove-background', app=app, **kwargs)
            self.assertEqual(status, 413); self.assertEqual(d['code'], 'REQUEST_TOO_LARGE')
        self.assertEqual(self.runtime.calls, [])

    def test_response_size_via_route(self):
        app = create_app(Settings(max_response_bytes=65537), self.runtime)
        status, d, _ = self.post('/remove-background', app=app)
        self.assertEqual(status, 413); self.assertEqual(d['code'], 'RESPONSE_TOO_LARGE')

    def test_internal_error_and_logs_sanitized(self):
        self.runtime.failure = True
        with self.assertLogs('faber.inference', logging.INFO) as logs:
            status, d, _ = self.post('/segment', [('point_x','3'),('point_y','3')])
        self.assertEqual(status, 200); self.assertEqual(d['code'], 'INFERENCE_FAILED')
        text = json.dumps(d) + ''.join(logs.output)
        for sensitive in ['SECRET','/private','token=','private-name','point_x','mask_png_base64']:
            self.assertNotIn(sensitive, text)
        self.assertIn('latency_ms', text)

    def test_health_ready_without_birefnet(self):
        for path in ['/health', '/ready']:
            status, d, _ = asyncio.run(asgi_request(self.app, path))
            self.assertEqual(status, 200); self.assertFalse(d['birefnet_loaded'])
        self.runtime.sam_ready = False
        status, d, _ = asyncio.run(asgi_request(self.app, '/ready'))
        self.assertEqual(status, 503); self.assertFalse(d['ok'])
        self.assertEqual(asyncio.run(asgi_request(self.app, '/health'))[0], 200)

    def test_local_cors(self):
        _, _, headers = asyncio.run(asgi_request(self.app, '/health', extra_headers=[(b'origin',b'http://localhost:3000')]))
        self.assertEqual(headers[b'access-control-allow-origin'], b'http://localhost:3000')
        app = create_app(Settings(cors_origins=()), self.runtime)
        _, _, headers = asyncio.run(asgi_request(app, '/health', extra_headers=[(b'origin',b'https://external.test')]))
        self.assertNotIn(b'access-control-allow-origin', headers)

    def test_lifespan_fake_start(self):
        self.runtime.sam_ready = False
        async def exercise():
            async with self.app.router.lifespan_context(self.app):
                self.assertTrue(self.runtime.sam_ready)
        asyncio.run(exercise())

    def test_lifespan_failure_sanitized(self):
        def failure(): raise RuntimeError('SECRET /private/cache')
        self.runtime.start = failure
        async def exercise():
            async with self.app.router.lifespan_context(self.app): pass
        with self.assertLogs('faber.inference', logging.ERROR) as logs:
            with self.assertRaisesRegex(RuntimeError, '^MODEL_STARTUP_FAILED$'): asyncio.run(exercise())
        self.assertNotIn('SECRET', ''.join(logs.output))


class EntrypointTest(unittest.TestCase):
    def test_uvicorn_error_cannot_expose_traceback(self):
        from run_server import ServerErrorFormatter
        record = logging.LogRecord('uvicorn.error', logging.ERROR, '/private/model.py', 1,
                                   'SECRET traceback /private/token', (), None)
        output = ServerErrorFormatter().format(record)
        self.assertEqual(json.loads(output)['code'], 'SERVER_RUNTIME_ERROR')
        self.assertNotIn('SECRET', output)
        self.assertNotIn('/private', output)

    def test_entrypoint_port_host_one_worker(self):
        import types
        from run_server import main
        calls = []
        with patch.dict(sys.modules, {'uvicorn': types.SimpleNamespace(run=lambda *a, **k: calls.append(k))}):
            with patch('run_server.Settings.from_env', return_value=Settings(port=9090)):
                self.assertEqual(main(), 0)
        self.assertEqual(calls[0]['host'], '0.0.0.0')
        self.assertEqual(calls[0]['port'], 9090)
        self.assertEqual(calls[0]['workers'], 1)
        self.assertFalse(calls[0]['access_log'])


if __name__ == '__main__':
    unittest.main()
