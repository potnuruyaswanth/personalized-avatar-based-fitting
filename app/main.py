from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from app.tryon import TryOnConfigError, TryOnInferenceError, generate_try_on, reload_runtime, runtime_info

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"

app = FastAPI(title="Personalized Avatar-Based Fitting", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/assets", StaticFiles(directory=FRONTEND_DIR), name="assets")


@app.get("/", include_in_schema=False)
def read_index() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")


@app.post("/api/try-on")
async def try_on(person: UploadFile = File(...), cloth: UploadFile = File(...)) -> Response:
    image_types = {"image/jpeg", "image/png", "image/webp"}

    if person.content_type not in image_types:
        raise HTTPException(status_code=400, detail="Invalid person image type")
    if cloth.content_type not in image_types:
        raise HTTPException(status_code=400, detail="Invalid cloth image type")

    person_bytes = await person.read()
    cloth_bytes = await cloth.read()

    if not person_bytes or not cloth_bytes:
        raise HTTPException(status_code=400, detail="Both images are required")

    try:
        result_image = generate_try_on(person_bytes, cloth_bytes)
    except TryOnConfigError as exc:
        raise HTTPException(status_code=500, detail=f"Try-on config error: {exc}") from exc
    except TryOnInferenceError as exc:
        raise HTTPException(status_code=500, detail=f"Try-on model error: {exc}") from exc
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"Try-on failed: {exc}") from exc

    return Response(content=result_image, media_type="image/png")


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/model-info")
def get_model_info() -> dict:
    return runtime_info()


@app.post("/api/model-reload")
def post_model_reload() -> dict:
    return reload_runtime()
