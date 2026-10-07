"""Non-secret server settings; limits may be lowered, never raised past safety ceilings."""
from dataclasses import dataclass
from pathlib import Path
import os
import re

MIB = 1024 * 1024
BIREFNET_REVISION = "e2bf8e4460fc8fa32bba5ea4d94b3233d367b0e4"
BASE_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Settings:
    port: int = 8080
    sam_checkpoint: Path = BASE_DIR / "models/sam_vit_b_01ec64.pth"
    birefnet_path: str = ""
    birefnet_revision: str = BIREFNET_REVISION
    max_upload_bytes: int = 20 * MIB
    max_request_bytes: int = 21 * MIB
    max_pixels: int = 24_000_000
    max_width: int = 8192
    max_height: int = 8192
    max_response_bytes: int = 28 * MIB
    max_points: int = 64  # Per positive/negative array.
    max_prompt_bytes: int = 16 * 1024
    torch_threads: int = 2
    log_level: str = "INFO"
    cors_origins: tuple[str, ...] = ("http://localhost:3000", "http://127.0.0.1:3000")

    def __post_init__(self):
        ceilings = {"port": 65535, "max_upload_bytes": 20 * MIB,
                    "max_request_bytes": 21 * MIB, "max_pixels": 24_000_000,
                    "max_width": 8192, "max_height": 8192,
                    "max_response_bytes": 28 * MIB, "max_points": 64,
                    "max_prompt_bytes": 16 * 1024, "torch_threads": 16}
        if any(type(getattr(self, key)) is not int or not 1 <= getattr(self, key) <= ceiling
               for key, ceiling in ceilings.items()):
            raise ValueError("INVALID_SERVER_CONFIGURATION")
        if self.max_request_bytes <= self.max_upload_bytes or self.max_response_bytes <= 65536:
            raise ValueError("INVALID_SERVER_CONFIGURATION")
        if not re.fullmatch(r"[0-9a-f]{40}", self.birefnet_revision):
            raise ValueError("INVALID_MODEL_REVISION")
        if self.log_level not in {"INFO", "WARNING", "ERROR"}:
            raise ValueError("INVALID_LOG_LEVEL")
        if any(not re.fullmatch(r"https?://[^/\s*]+", origin) for origin in self.cors_origins):
            raise ValueError("INVALID_CORS_ORIGIN")

    @classmethod
    def from_env(cls, env=None):
        env = os.environ if env is None else env
        defaults = cls()
        values = {}
        for name in ("port", "max_upload_bytes", "max_request_bytes", "max_pixels", "max_width",
                     "max_height", "max_response_bytes", "max_points", "max_prompt_bytes", "torch_threads"):
            try:
                values[name] = int(env.get(name.upper(), getattr(defaults, name)))
            except (TypeError, ValueError):
                raise ValueError("INVALID_SERVER_CONFIGURATION") from None
        values.update(sam_checkpoint=Path(env.get("SAM_CHECKPOINT_PATH", str(defaults.sam_checkpoint))),
                      birefnet_path=env.get("BIREFNET_MODEL_PATH", ""),
                      birefnet_revision=env.get("BIREFNET_REVISION", BIREFNET_REVISION),
                      log_level=env.get("LOG_LEVEL", "INFO").upper())
        if "CORS_ORIGINS" in env:
            values["cors_origins"] = tuple(x.strip() for x in env["CORS_ORIGINS"].split(",") if x.strip())
        return cls(**values)
