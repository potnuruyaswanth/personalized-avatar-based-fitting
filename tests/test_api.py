from __future__ import annotations

from io import BytesIO

from PIL import Image
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def _img_bytes(color: tuple[int, int, int], size: tuple[int, int]) -> bytes:
    image = Image.new("RGB", size, color)
    buf = BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def test_health_endpoint() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_model_info_endpoint() -> None:
    response = client.get("/api/model-info")
    assert response.status_code == 200
    payload = response.json()
    assert "active_backend" in payload
    assert "backends" in payload


def test_model_reload_endpoint() -> None:
    response = client.post("/api/model-reload")
    assert response.status_code == 200
    payload = response.json()
    assert "active_backend" in payload


def test_try_on_endpoint_returns_png() -> None:
    files = {
        "person": ("person.png", _img_bytes((220, 210, 200), (720, 1024)), "image/png"),
        "cloth": ("cloth.png", _img_bytes((15, 98, 182), (450, 520)), "image/png"),
    }

    response = client.post("/api/try-on", files=files)
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("image/png")
    assert len(response.content) > 0
