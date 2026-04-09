from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from io import BytesIO
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Lock
from typing import Any

import yaml
from PIL import Image, ImageEnhance, ImageFilter, ImageOps


BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = BASE_DIR / "configs" / "tryon.yaml"


class TryOnConfigError(RuntimeError):
    """Raised when the try-on engine config is invalid."""


class TryOnInferenceError(RuntimeError):
    """Raised when selected model inference fails."""


def _build_mask_from_cloth(cloth_rgba: Image.Image) -> Image.Image:
    """Create a soft transparency mask for the cloth image."""
    alpha = cloth_rgba.split()[-1]
    if alpha.getbbox() is not None:
        return alpha.filter(ImageFilter.GaussianBlur(radius=1.4))

    gray = ImageOps.grayscale(cloth_rgba)
    inv = ImageOps.invert(gray)
    boosted = ImageEnhance.Contrast(inv).enhance(2.0)
    return boosted.filter(ImageFilter.GaussianBlur(radius=2.2))


def _generate_baseline_try_on(person_bytes: bytes, cloth_bytes: bytes) -> bytes:
    """Generate a baseline virtual try-on composition using geometric placement."""
    person = Image.open(BytesIO(person_bytes)).convert("RGBA")
    cloth = Image.open(BytesIO(cloth_bytes)).convert("RGBA")

    person_w, person_h = person.size

    # Heuristic placement tuned for front-facing full-body shots.
    target_w = max(int(person_w * 0.44), 80)
    ratio = target_w / max(cloth.width, 1)
    target_h = int(cloth.height * ratio)
    target_h = min(target_h, int(person_h * 0.52))

    cloth_resized = cloth.resize((target_w, target_h), Image.Resampling.LANCZOS)
    mask = _build_mask_from_cloth(cloth_resized)

    x = (person_w - target_w) // 2
    y = int(person_h * 0.24)

    canvas = person.copy()
    canvas.paste(cloth_resized, (x, y), mask)

    out = BytesIO()
    canvas.convert("RGB").save(out, format="PNG", optimize=True)
    return out.getvalue()


@dataclass
class CommandBackend:
    """Runs CP-VTON / VITON-HD inference by executing configured command-line scripts."""

    name: str
    cfg: dict[str, Any]

    def __post_init__(self) -> None:
        self.enabled = bool(self.cfg.get("enabled", False))
        self.repo_dir = (BASE_DIR / self.cfg.get("repo_dir", "")).resolve()
        self.checkpoint_path = (BASE_DIR / self.cfg.get("checkpoint_path", "")).resolve()
        self.python_executable = self.cfg.get("python_executable") or sys.executable
        self.inference_command = self.cfg.get("inference_command", [])
        self.output_filename = self.cfg.get("output_filename", "result.png")
        self.custom_placeholders = self.cfg.get("placeholders", {})
        self.env = self.cfg.get("env", {})

    @property
    def is_ready(self) -> bool:
        return self.enabled and self.repo_dir.exists() and isinstance(self.inference_command, list)

    def run(self, person_bytes: bytes, cloth_bytes: bytes) -> bytes:
        if not self.is_ready:
            raise TryOnInferenceError(
                f"Backend '{self.name}' is not ready. Check enabled flag, repo_dir, and inference_command."
            )

        with TemporaryDirectory(prefix=f"{self.name}-") as tmp_dir:
            tmp_path = Path(tmp_dir)
            person_path = tmp_path / "person.png"
            cloth_path = tmp_path / "cloth.png"
            output_path = tmp_path / self.output_filename

            person_path.write_bytes(person_bytes)
            cloth_path.write_bytes(cloth_bytes)

            placeholders = {
                "python": self.python_executable,
                "person": str(person_path),
                "cloth": str(cloth_path),
                "output": str(output_path),
                "checkpoint": str(self.checkpoint_path),
            }

            if isinstance(self.custom_placeholders, dict):
                for key, value in self.custom_placeholders.items():
                    if not isinstance(key, str):
                        continue
                    if isinstance(value, str):
                        resolved = (BASE_DIR / value).resolve() if not Path(value).is_absolute() else Path(value)
                        placeholders[key] = str(resolved)
                    else:
                        placeholders[key] = str(value)

            process_env = None
            if isinstance(self.env, dict):
                process_env = dict(os.environ)
                process_env.update({str(k): str(v) for k, v in self.env.items()})

            command = [str(token).format(**placeholders) for token in self.inference_command]

            completed = subprocess.run(
                command,
                cwd=str(self.repo_dir),
                env=process_env,
                capture_output=True,
                text=True,
                check=False,
            )

            if completed.returncode != 0:
                error_message = completed.stderr.strip() or completed.stdout.strip() or "unknown error"
                raise TryOnInferenceError(
                    f"{self.name} inference failed (exit {completed.returncode}): {error_message}"
                )

            if not output_path.exists():
                raise TryOnInferenceError(
                    f"{self.name} inference completed but output file was not produced at {output_path}"
                )

            return output_path.read_bytes()


class TryOnRuntime:
    """Holds runtime config and dispatches inference to selected backend."""

    def __init__(self, config_path: Path = DEFAULT_CONFIG_PATH) -> None:
        self.config_path = config_path
        self.config = self._load_config(config_path)
        self.active_backend = str(self.config.get("active_backend", "baseline")).strip().lower()
        self.allow_fallback = bool(self.config.get("allow_fallback_to_baseline", True))

        backend_cfg = self.config.get("backends", {})
        self.backends: dict[str, CommandBackend] = {
            "cp_vton": CommandBackend("cp_vton", backend_cfg.get("cp_vton", {})),
            "viton_hd": CommandBackend("viton_hd", backend_cfg.get("viton_hd", {})),
        }

    def _load_config(self, config_path: Path) -> dict[str, Any]:
        if not config_path.exists():
            raise TryOnConfigError(f"Missing try-on config file: {config_path}")

        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise TryOnConfigError("Invalid try-on config format: expected mapping/object")
        return raw

    def generate(self, person_bytes: bytes, cloth_bytes: bytes) -> bytes:
        if self.active_backend == "baseline":
            return _generate_baseline_try_on(person_bytes, cloth_bytes)

        backend = self.backends.get(self.active_backend)
        if backend is None:
            raise TryOnConfigError(f"Unsupported active_backend '{self.active_backend}'")

        try:
            return backend.run(person_bytes, cloth_bytes)
        except TryOnInferenceError:
            if not self.allow_fallback:
                raise
            return _generate_baseline_try_on(person_bytes, cloth_bytes)

    def info(self) -> dict[str, Any]:
        return {
            "config_path": str(self.config_path),
            "active_backend": self.active_backend,
            "allow_fallback_to_baseline": self.allow_fallback,
            "backends": {
                "cp_vton": {
                    "enabled": self.backends["cp_vton"].enabled,
                    "repo_dir": str(self.backends["cp_vton"].repo_dir),
                    "is_ready": self.backends["cp_vton"].is_ready,
                },
                "viton_hd": {
                    "enabled": self.backends["viton_hd"].enabled,
                    "repo_dir": str(self.backends["viton_hd"].repo_dir),
                    "is_ready": self.backends["viton_hd"].is_ready,
                },
            },
        }


_RUNTIME_LOCK = Lock()
_RUNTIME: TryOnRuntime | None = None


def _get_runtime() -> TryOnRuntime:
    global _RUNTIME
    if _RUNTIME is None:
        with _RUNTIME_LOCK:
            if _RUNTIME is None:
                _RUNTIME = TryOnRuntime()
    return _RUNTIME


def reload_runtime() -> dict[str, Any]:
    """Reload runtime config without restarting the API service."""
    global _RUNTIME
    with _RUNTIME_LOCK:
        _RUNTIME = TryOnRuntime()
    return _RUNTIME.info()


def runtime_info() -> dict[str, Any]:
    return _get_runtime().info()


def generate_try_on(person_bytes: bytes, cloth_bytes: bytes) -> bytes:
    return _get_runtime().generate(person_bytes, cloth_bytes)
