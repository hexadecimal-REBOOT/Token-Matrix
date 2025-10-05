"""FastAPI backend for converting Video-Depth-Anything checkpoints to ONNX."""

from __future__ import annotations

import asyncio
import logging
import shutil
import tempfile
from pathlib import Path
from typing import Iterable, List

import torch
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from tools.export_vda_onnx import load_model

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

app = FastAPI(title="Video-Depth-Anything Converter", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
async def health() -> dict[str, str]:
    """Simple readiness probe."""
    return {"status": "ok"}


def _sanitize_path(filename: str) -> Path:
    cleaned = Path(filename)
    safe_parts = [part for part in cleaned.parts if part not in ("", ".", "..")]
    if not safe_parts:
        safe_parts = [cleaned.name or "file.bin"]
    return Path(*safe_parts)


def _find_checkpoint(root: Path) -> Path | None:
    for pattern in ("*.pth", "*.pt", "*.ckpt"):
        match = next(root.rglob(pattern), None)
        if match:
            return match
    return None


def _detect_repo_root(root: Path) -> Path:
    candidates: Iterable[Path] = root.rglob("video_depth_anything")
    for candidate in candidates:
        if candidate.is_dir():
            return candidate.parent
    return root


async def _save_uploads(files: List[UploadFile]) -> Path:
    temp_dir = Path(tempfile.mkdtemp(prefix="vda_upload_"))
    for upload in files:
        relative = _sanitize_path(upload.filename)
        destination = temp_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)

        logger.info("Saving %s", relative)
        with destination.open("wb") as buffer:
            while True:
                chunk = await upload.read(1024 * 1024)
                if not chunk:
                    break
                buffer.write(chunk)
        await upload.close()
    return temp_dir


def _build_dummy_input(batch: int, seq: int, height: int, width: int) -> torch.Tensor:
    if height % 14 or width % 14:
        raise HTTPException(status_code=400, detail="Input dimensions must be multiples of 14.")
    try:
        dummy = torch.randn(batch, seq, 3, height, width, dtype=torch.float32)
    except RuntimeError as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=500, detail=f"Failed to allocate dummy tensor: {exc}") from exc
    return dummy


@app.post("/api/convert")
async def convert_checkpoint(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    encoder: str = Form("vitl"),
    metric: bool = Form(False),
    input_height: int = Form(518),
    input_width: int = Form(518),
    sequence_length: int = Form(32),
    batch_size: int = Form(1),
    opset: int = Form(17),
    dynamic_axes: bool = Form(True),
) -> FileResponse | JSONResponse:
    if not files:
        raise HTTPException(status_code=400, detail="No files were uploaded.")

    upload_dir = await _save_uploads(files)
    checkpoint_path = _find_checkpoint(upload_dir)
    if not checkpoint_path:
        shutil.rmtree(upload_dir, ignore_errors=True)
        raise HTTPException(status_code=400, detail="Could not locate a checkpoint (.pth/.pt/.ckpt) in the upload.")

    repo_root = _detect_repo_root(upload_dir)
    logger.info("Detected repository root: %s", repo_root)

    try:
        model = load_model(repo_root, encoder=encoder, metric=metric)
    except SystemExit as exc:
        shutil.rmtree(upload_dir, ignore_errors=True)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - defensive
        shutil.rmtree(upload_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"Failed to construct model: {exc}") from exc

    try:
        state = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(state, strict=True)
        model.eval().to(torch.float32)
    except Exception as exc:
        shutil.rmtree(upload_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"Failed to load checkpoint: {exc}") from exc

    dummy = _build_dummy_input(batch_size, sequence_length, input_height, input_width)

    export_path = upload_dir / "video_depth_anything.onnx"
    dynamic_axes_map = (
        {
            "frames": {0: "batch", 1: "time", 3: "height", 4: "width"},
            "depth": {0: "batch", 1: "time", 2: "height", 3: "width"},
        }
        if dynamic_axes
        else None
    )

    try:
        with torch.no_grad():
            torch.onnx.export(
                model,
                dummy,
                str(export_path),
                input_names=["frames"],
                output_names=["depth"],
                dynamic_axes=dynamic_axes_map,
                opset_version=opset,
                do_constant_folding=True,
            )
    except Exception as exc:
        shutil.rmtree(upload_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"ONNX export failed: {exc}") from exc

    logger.info("Export complete: %s", export_path)
    background_tasks.add_task(shutil.rmtree, upload_dir, True)

    headers = {"x-export-filename": export_path.name}
    return FileResponse(
        path=export_path,
        media_type="application/octet-stream",
        filename=export_path.name,
        headers=headers,
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(_, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


# Ensure the event loop policy works when running under uvicorn on Windows.
try:  # pragma: no cover - platform-specific safeguard
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
except AttributeError:
    pass
