"""Model lifecycle and serialized inference; importing this module loads no weights."""
from threading import Lock
from pathlib import Path
from server_safety import PublicError

BIREFNET_MODEL_ID = "ZhengPeng7/BiRefNet"


class LazyResource:
    def __init__(self, loader):
        self.loader = loader
        self._value = None
        self._lock = Lock()

    @property
    def ready(self):
        return self._value is not None

    def get(self):
        with self._lock:
            if self._value is None:
                self._value = self.loader()
            return self._value


class InferenceRuntime:
    def __init__(self, settings, sam_loader=None, birefnet_loader=None):
        self.settings = settings
        self.device = "uninitialized"
        self.predictor = None
        # One process owns its own models and lock. Container entrypoint enforces one worker.
        self._inference_lock = Lock()
        self._birefnet = LazyResource(birefnet_loader or self._load_birefnet)
        self._sam_loader = sam_loader or self._load_sam

    @property
    def sam_ready(self):
        return self.predictor is not None

    @property
    def birefnet_loaded(self):
        return self._birefnet.ready

    def start(self):
        with self._inference_lock:
            if self.predictor is None:
                self.predictor = self._sam_loader()

    def _load_sam(self):
        if not self.settings.sam_checkpoint.is_file():
            raise RuntimeError("SAM_CHECKPOINT_MISSING")
        import torch
        from segment_anything import SamPredictor, sam_model_registry
        torch.set_num_threads(self.settings.torch_threads)
        torch.set_num_interop_threads(1)
        self.device = ("cuda" if torch.cuda.is_available() else
                       "mps" if torch.backends.mps.is_available() else "cpu")
        model = sam_model_registry["vit_b"](checkpoint=str(self.settings.sam_checkpoint))
        model.to(device=self.device)
        model.eval()
        return SamPredictor(model)

    def _load_birefnet(self):
        from transformers import AutoModelForImageSegmentation
        source = self.settings.birefnet_path or BIREFNET_MODEL_ID
        if self.settings.birefnet_path and not Path(source).is_dir():
            raise PublicError("MODEL_UNAVAILABLE")
        # Offline by default for native + container: only an existing pinned cache or local bundle.
        model = AutoModelForImageSegmentation.from_pretrained(
            source, trust_remote_code=True, revision=self.settings.birefnet_revision,
            code_revision=self.settings.birefnet_revision, local_files_only=True,
        )
        model.eval()
        model.to(device=self.device)
        return model

    def segment(self, image, **kwargs):
        with self._inference_lock:
            if self.predictor is None:
                raise PublicError("MODEL_UNAVAILABLE")
            self.predictor.set_image(image)
            # Includes all predictor state accesses; never unlock between set_image and predict.
            return self.predictor.predict(**kwargs)

    def remove_background(self, image):
        with self._inference_lock:
            try:
                model = self._birefnet.get()
            except Exception:
                raise PublicError("MODEL_UNAVAILABLE") from None
            import numpy as np
            import torch
            from torchvision import transforms
            from PIL import Image
            from alpha_preservation import preserve_alpha
            transform = transforms.Compose([
                transforms.Resize((1024, 1024)), transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
            tensor = transform(image.convert("RGB")).unsqueeze(0).to(self.device)
            with torch.inference_mode():
                prediction = model(tensor)[-1].sigmoid().detach().cpu()[0].squeeze()
            small = (prediction.clamp(0, 1).numpy() * 255.0).astype(np.uint8)
            mask = Image.fromarray(small).resize(image.size, Image.Resampling.LANCZOS)
            mask, cutout = preserve_alpha(image, mask)
            return np.array(mask, dtype=np.uint8), cutout
