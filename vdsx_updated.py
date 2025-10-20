#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VDSX Updated Pipeline (RGB + Segm + Depth + Detection + OCR -> Context Map -> Vision Tokens -> ZMQ)
- Runs on Raspberry Pi (camera hub) or Jetson Orin Nano/Super (consumer)
- Graceful fallbacks: if YOLO-World or Tesseract/PaddleOCR not present, we still run
- Streams token packets over ZeroMQ PUB/SUB (JSON + msgpack payload)
- Minimal external deps: opencv-python, numpy, pyzmq, msgpack, (optional) ultralytics, pytesseract/paddleocr

Author: VDSX
Date: 2025-10-17
"""

import os
import sys
import time
import json
import zlib
import math
import uuid
import queue
import base64
import struct
import msgpack
import argparse
import threading
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional

import numpy as np
import cv2

# -------------------------
# Optional imports
# -------------------------
HAS_ULTRALYTICS = False
try:
    from ultralytics import YOLOWorld
    HAS_ULTRALYTICS = True
except Exception:
    pass

HAS_TESSERACT = False
try:
    import pytesseract
    HAS_TESSERACT = True
except Exception:
    pass

HAS_PADDLEOCR = False
try:
    from paddleocr import PaddleOCR
    HAS_PADDLEOCR = True
except Exception:
    pass

import zmq


# ============================================
# Configuration dataclasses
# ============================================

@dataclass
class CameraConfig:
    device: int = 0                   # OpenCV index
    width: int = 1280
    height: int = 720
    fps: int = 10
    libcamera_gstreamer: Optional[str] = None  # If provided, overrides OpenCV index with pipeline string


@dataclass
class DetectionConfig:
    enabled: bool = True
    model: str = "yolov8s-world"      # requires ultralytics>=8.1
    conf: float = 0.30
    iou: float = 0.45
    classes: List[str] = field(default_factory=lambda: [
        "person", "pedestrian", "car", "truck", "bus",
        "bicycle", "motorcycle", "traffic light", "stop sign",
        "construction cone", "road barrier", "dog", "pothole", "road debris"
    ])
    device: str = "cuda:0"            # or 'cpu'


@dataclass
class OCRConfig:
    enabled: bool = True
    use_paddleocr: bool = False       # if True and PaddleOCR available, use it; else Tesseract if available
    paddle_lang: str = "en"
    tesseract_oem_psm: str = "--oem 3 --psm 6"  # adjust for scene text
    min_confidence: float = 0.55


@dataclass
class DepthConfig:
    enabled: bool = False             # If you have disparity → depth, set True & feed map
    baseline_m: float = 0.18          # not used unless depth provided
    fx: float = 700.0                 # not used unless depth provided


@dataclass
class SegmentationConfig:
    enabled: bool = False             # placeholder/simple segm mask
    # You can swap with a real segmentation model later


@dataclass
class TokenConfig:
    patch: int = 16                   # patch size for tokenization
    dim: int = 64                     # compressed token dim (SVD/avg pooling synthetic)
    keep_ratio: float = 0.15          # keep top-K patches by variance/energy
    compress: bool = True             # zlib compress payload


@dataclass
class StreamConfig:
    role: str = "producer"            # 'producer' (camera hub) or 'consumer' (analytics)
    zmq_bind: str = "tcp://0.0.0.0:5557"    # for producer: bind here
    zmq_connect: str = "tcp://127.0.0.1:5557"  # for consumer: connect here
    topic: str = "vdsx.tokens"
    send_raw_thumbnails: bool = False
    thumbnail_size: Tuple[int, int] = (320, 180)


@dataclass
class OverlayConfig:
    show_depth: bool = True
    show_boxes: bool = True
    show_text: bool = True
    show_fps: bool = True
    font_scale: float = 0.6
    thickness: int = 2


@dataclass
class VDSXConfig:
    camera: CameraConfig = CameraConfig()
    detect: DetectionConfig = DetectionConfig()
    ocr: OCRConfig = OCRConfig()
    depth: DepthConfig = DepthConfig()
    segm: SegmentationConfig = SegmentationConfig()
    tokens: TokenConfig = TokenConfig()
    stream: StreamConfig = StreamConfig()
    overlay: OverlayConfig = OverlayConfig()
    context_size: Tuple[int, int] = (640, 360)  # composite canvas size for tokenization


# ============================================
# Video source
# ============================================

class VideoSource:
    def __init__(self, cfg: CameraConfig):
        self.cfg = cfg
        self.cap = None

    def open(self) -> bool:
        if self.cfg.libcamera_gstreamer:
            pipeline = self.cfg.libcamera_gstreamer
            self.cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        else:
            self.cap = cv2.VideoCapture(self.cfg.device)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.cfg.fps)

        ok = bool(self.cap and self.cap.isOpened())
        if not ok:
            print("[camera] ERROR: could not open capture")
        return ok

    def read(self) -> Optional[np.ndarray]:
        if not self.cap:
            return None
        ret, frame = self.cap.read()
        return frame if ret else None

    def close(self):
        try:
            if self.cap:
                self.cap.release()
        except Exception:
            pass


# ============================================
# Optional segmentation placeholder
# ============================================

class SimpleSegmenter:
    """Very light placeholder: edge-based pseudo mask to visualize context.
    Replace later with a real segmentation model."""
    def __init__(self):
        pass

    def run(self, frame_bgr: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 60, 120)
        mask = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
        mask = (mask > 0).astype(np.uint8) * 255
        mask_bgr = cv2.applyColorMap(mask, cv2.COLORMAP_TURBO)
        return mask_bgr


# ============================================
# Object detection (YOLO-World, optional)
# ============================================

class YOLOWorldDetector:
    def __init__(self, cfg: DetectionConfig):
        self.cfg = cfg
        self.ok = False
        if self.cfg.enabled and HAS_ULTRALYTICS:
            try:
                self.model = YOLOWorld(self.cfg.model)
                self.model.to(self.cfg.device if "cuda" in self.cfg.device else "cpu")
                self.model.set_classes(self.cfg.classes)
                self.ok = True
                print(f"[detect] YOLO-World ready: {self.cfg.model} | classes: {len(self.cfg.classes)}")
            except Exception as e:
                print(f"[detect] WARN: failed to init YOLO-World: {e}")
        else:
            print("[detect] YOLO-World disabled or ultralytics not installed.")

    def run(self, frame_bgr: np.ndarray) -> List[Dict]:
        if not self.ok:
            return []
        res = self.model.predict(frame_bgr, conf=self.cfg.conf, iou=self.cfg.iou, verbose=False)[0]
        dets = []
        for box in res.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            cls_name = self.cfg.classes[cls_id] if 0 <= cls_id < len(self.cfg.classes) else f"cls_{cls_id}"
            dets.append({
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "conf": conf,
                "name": cls_name,
                "cls": cls_id
            })
        return dets


# ============================================
# OCR (Tesseract or PaddleOCR)
# ============================================

class OCREngine:
    def __init__(self, cfg: OCRConfig):
        self.cfg = cfg
        self.mode = "none"
        self.ocr = None
        if not self.cfg.enabled:
            print("[ocr] disabled")
            return
        if self.cfg.use_paddleocr and HAS_PADDLEOCR:
            try:
                self.ocr = PaddleOCR(use_angle_cls=True, lang=self.cfg.paddle_lang, show_log=False)
                self.mode = "paddle"
                print("[ocr] PaddleOCR ready")
                return
            except Exception as e:
                print(f"[ocr] WARN PaddleOCR init failed: {e}")

        if HAS_TESSERACT:
            self.mode = "tesseract"
            print("[ocr] Tesseract ready")
        else:
            print("[ocr] WARN no OCR engine available")

    def run(self, frame_bgr: np.ndarray, rois: Optional[List[Tuple[int, int, int, int]]] = None) -> List[Dict]:
        results = []
        if self.mode == "none":
            return results

        h, w = frame_bgr.shape[:2]
        if not rois:
            # coarsely tile or just full frame
            rois = [(0, 0, w, h)]

        for (x1, y1, x2, y2) in rois:
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(0, min(x2, w - 1))
            y2 = max(0, min(y2, h - 1))
            if x2 <= x1 or y2 <= y1:
                continue
            crop = frame_bgr[y1:y2, x1:x2]

            if self.mode == "paddle":
                try:
                    out = self.ocr.ocr(crop, cls=True)
                    for line in out[0] if out else []:
                        txt = line[1][0]
                        conf = float(line[1][1])
                        if conf >= self.cfg.min_confidence:
                            results.append({"text": txt, "conf": conf, "bbox": [x1, y1, x2, y2]})
                except Exception:
                    pass

            elif self.mode == "tesseract":
                try:
                    config = self.cfg.tesseract_oem_psm
                    raw = pytesseract.image_to_data(crop, config=config, output_type=pytesseract.Output.DICT)
                    n = len(raw.get("text", []))
                    for i in range(n):
                        txt = raw["text"][i].strip()
                        conf = float(raw["conf"][i]) if raw["conf"][i] not in ["-1", ""] else 0.0
                        if txt and conf >= (self.cfg.min_confidence * 100.0):  # tesseract conf is 0..100
                            rx, ry, rw_, rh_ = raw["left"][i], raw["top"][i], raw["width"][i], raw["height"][i]
                            results.append({
                                "text": txt,
                                "conf": conf / 100.0,
                                "bbox": [x1 + rx, y1 + ry, x1 + rx + rw_, y1 + ry + rh_]
                            })
                except Exception:
                    pass

        return results


# ============================================
# Context composer (combine RGB + segm + depth + boxes + OCR into a single canvas)
# ============================================

class ContextComposer:
    def __init__(self, out_size: Tuple[int, int], overlay: OverlayConfig):
        self.w, self.h = out_size
        self.overlay = overlay

    @staticmethod
    def _draw_boxes(canvas: np.ndarray, detections: List[Dict]):
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            name = det.get("name", "obj")
            conf = det.get("conf", 0.0)
            color = (0, 255, 0)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
            label = f"{name} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(canvas, (x1, y1 - th - 6), (x1 + tw, y1), color, -1)
            cv2.putText(canvas, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    @staticmethod
    def _draw_texts(canvas: np.ndarray, texts: List[Dict]):
        for t in texts:
            x1, y1, x2, y2 = t["bbox"]
            txt = t.get("text", "")
            conf = t.get("conf", 0.0)
            color = (255, 0, 255)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 1)
            label = f"{txt[:24]} ({conf:.2f})"
            cv2.putText(canvas, label, (x1, max(20, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    def compose(self,
                rgb: np.ndarray,
                segm: Optional[np.ndarray],
                depth: Optional[np.ndarray],
                detections: List[Dict],
                ocr_texts: List[Dict],
                fps: float = 0.0) -> np.ndarray:
        # Resize RGB to canvas
        canvas = cv2.resize(rgb, (self.w, self.h), interpolation=cv2.INTER_AREA)

        # Segmentation overlay (if provided)
        if segm is not None and self.overlay.show_depth:
            segm_resized = cv2.resize(segm, (self.w // 3, self.h // 3))
            canvas[10:10 + segm_resized.shape[0], 10:10 + segm_resized.shape[1]] = segm_resized

        # Depth overlay (if provided as heatmap)
        if depth is not None and self.overlay.show_depth:
            depth_norm = depth.copy().astype(np.float32)
            dmin, dmax = np.percentile(depth_norm[depth_norm > 0], [5, 95]) if np.any(depth_norm > 0) else (0, 1)
            depth_norm = np.clip((depth_norm - dmin) / (dmax - dmin + 1e-6), 0, 1)
            depth_vis = (depth_norm * 255.0).astype(np.uint8)
            depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_PLASMA)
            depth_vis = cv2.resize(depth_vis, (self.w // 3, self.h // 3))
            canvas[10:10 + depth_vis.shape[0], 20 + (self.w // 3):20 + (self.w // 3) + depth_vis.shape[1]] = depth_vis

        # Draw detections
        if self.overlay.show_boxes and detections:
            self._draw_boxes(canvas, detections)

        # Draw OCR
        if self.overlay.show_text and ocr_texts:
            self._draw_texts(canvas, ocr_texts)

        # FPS/TS
        if self.overlay.show_fps:
            ts = time.strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(canvas, f"{ts} | FPS {fps:.1f}", (10, self.h - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        return canvas


# ============================================
# Tokenizer (patchify + simple feature reduction)
# ============================================

class VisionTokenizer:
    def __init__(self, cfg: TokenConfig):
        self.cfg = cfg

    def _patchify(self, img: np.ndarray) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
        h, w = img.shape[:2]
        p = self.cfg.patch
        # center crop to multiple of p
        h2 = (h // p) * p
        w2 = (w // p) * p
        y0 = (h - h2) // 2
        x0 = (w - w2) // 2
        crop = img[y0:y0 + h2, x0:x0 + w2]
        ph, pw = h2 // p, w2 // p

        patches = []
        coords = []
        for iy in range(ph):
            for ix in range(pw):
                y1, y2 = iy * p, (iy + 1) * p
                x1, x2 = ix * p, (ix + 1) * p
                patch = crop[y1:y2, x1:x2]
                patches.append(patch)
                coords.append((ix, iy))
        patches = np.stack(patches, axis=0)  # N, p, p, 3
        return patches, coords

    def _reduce(self, patches: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # simple stats-based feature: mean & std over channels → project to dim
        # (placeholder for a real deep encoder; fast and portable)
        N = patches.shape[0]
        feats = []
        energies = []
        for i in range(N):
            p = patches[i].astype(np.float32) / 255.0
            mu = p.mean(axis=(0, 1))               # 3
            sd = p.std(axis=(0, 1))                # 3
            flat = np.concatenate([mu, sd], axis=0)  # 6 dims
            feats.append(flat)
            # variance as energy
            energies.append(float(sd.mean()))
        X = np.stack(feats, axis=0)  # N, 6

        # Project to cfg.dim via a fixed random projection (deterministic seed)
        np.random.seed(42)
        W = np.random.randn(X.shape[1], self.cfg.dim).astype(np.float32) / math.sqrt(X.shape[1])
        Z = X @ W  # N, D
        energies = np.array(energies, dtype=np.float32)  # N
        return Z.astype(np.float32), energies

    def encode(self, canvas_bgr: np.ndarray) -> Dict:
        patches, coords = self._patchify(canvas_bgr)
        Z, energy = self._reduce(patches)
        # select top-K patches by energy
        K = max(1, int(len(energy) * self.cfg.keep_ratio))
        idx = np.argsort(-energy)[:K]
        sel_Z = Z[idx]
        sel_coords = [coords[i] for i in idx]

        payload = {
            "dim": self.cfg.dim,
            "patch": self.cfg.patch,
            "coords": sel_coords,               # list of (x, y)
            "tokens": sel_Z.astype(np.float32)  # K x dim
        }
        return payload

    @staticmethod
    def pack(payload: Dict, compress: bool = True) -> bytes:
        # convert tokens to bytes
        data = {
            "dim": payload["dim"],
            "patch": payload["patch"],
            "coords": payload["coords"],
            "tokens": payload["tokens"].tolist()
        }
        raw = msgpack.packb(data, use_bin_type=True)
        return zlib.compress(raw, level=6) if compress else raw


# ============================================
# ZMQ pub/sub
# ============================================

class ZMQPublisher:
    def __init__(self, bind_addr: str, topic: str):
        self.ctx = zmq.Context.instance()
        self.sock = self.ctx.socket(zmq.PUB)
        self.sock.setsockopt(zmq.SNDHWM, 10)
        self.sock.bind(bind_addr)
        self.topic = topic.encode("utf-8")
        print(f"[zmq] PUB bound: {bind_addr} | topic='{topic}'")

    def send(self, meta: Dict, payload: bytes):
        # meta as JSON (small) + binary payload
        self.sock.send_multipart([self.topic, json.dumps(meta).encode("utf-8"), payload])


class ZMQSubscriber:
    def __init__(self, connect_addr: str, topic: str, queue_size: int = 100):
        self.ctx = zmq.Context.instance()
        self.sock = self.ctx.socket(zmq.SUB)
        self.sock.setsockopt(zmq.RCVHWM, queue_size)
        self.sock.connect(connect_addr)
        self.sock.setsockopt_string(zmq.SUBSCRIBE, topic)
        print(f"[zmq] SUB connected: {connect_addr} | topic='{topic}'")

    def recv(self, block: bool = False, timeout_ms: int = 100):
        if block:
            parts = self.sock.recv_multipart()
            return parts
        else:
            if self.sock.poll(timeout=timeout_ms, flags=zmq.POLLIN):
                return self.sock.recv_multipart()
            return None


# ============================================
# Main App
# ============================================

class VDSXApp:
    def __init__(self, cfg: VDSXConfig):
        self.cfg = cfg
        self.cam = VideoSource(cfg.camera)
        self.seg = SimpleSegmenter() if cfg.segm.enabled else None
        self.det = YOLOWorldDetector(cfg.detect)
        self.ocr = OCREngine(cfg.ocr)
        self.composer = ContextComposer(cfg.context_size, cfg.overlay)
        self.tokenizer = VisionTokenizer(cfg.tokens)
        self.pub = None
        self.sub = None
        self.uuid = str(uuid.uuid4())[:8]
        self.fps_buf = []
        self.stop_flag = threading.Event()

        if cfg.stream.role == "producer":
            self.pub = ZMQPublisher(cfg.stream.zmq_bind, cfg.stream.topic)
        elif cfg.stream.role == "consumer":
            self.sub = ZMQSubscriber(cfg.stream.zmq_connect, cfg.stream.topic)

    # ------------- Utilities -------------
    def _fps(self, dt: float) -> float:
        if dt <= 0:
            return 0.0
        f = 1.0 / dt
        self.fps_buf.append(f)
        if len(self.fps_buf) > 30:
            self.fps_buf.pop(0)
        return sum(self.fps_buf) / max(1, len(self.fps_buf))

    # ------------- Producer loop -------------
    def run_producer(self):
        if not self.cam.open():
            return

        print("[vdsx] producer loop started")
        while not self.stop_flag.is_set():
            t0 = time.perf_counter()
            frame = self.cam.read()
            if frame is None:
                time.sleep(0.01)
                continue

            # Optional segmentation
            segm_vis = self.seg.run(frame) if self.seg else None

            # Optional depth (plug your depth map here if available)
            depth_map = None

            # Detection
            dets = self.det.run(frame) if self.cfg.detect.enabled else []

            # OCR (limit ROIs to detected signs/areas, else full frame)
            rois = [tuple(d["bbox"]) for d in dets] if dets else None
            texts = self.ocr.run(frame, rois=rois) if self.cfg.ocr.enabled else []

            dt = time.perf_counter() - t0
            fps = self._fps(dt)

            # Compose context canvas
            canvas = self.composer.compose(frame, segm_vis, depth_map, dets, texts, fps=fps)

            # Tokenize
            token_payload = self.tokenizer.encode(canvas)
            packed = self.tokenizer.pack(token_payload, compress=self.cfg.tokens.compress)

            # Meta
            meta = {
                "device": self.uuid,
                "ts": time.time(),
                "seq": int(time.time() * 1000),
                "cam_w": int(self.cfg.camera.width),
                "cam_h": int(self.cfg.camera.height),
                "ctx_w": int(self.cfg.context_size[0]),
                "ctx_h": int(self.cfg.context_size[1]),
                "fps": round(fps, 2),
                "dets": len(dets),
                "ocr": len(texts),
                "compress": self.cfg.tokens.compress
            }

            # Optional thumbnail for debugging
            if self.cfg.stream.send_raw_thumbnails:
                thumb = cv2.resize(canvas, self.cfg.stream.thumbnail_size)
                ok, jpg = cv2.imencode(".jpg", thumb, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                meta["thumb_b64"] = base64.b64encode(jpg.tobytes()).decode("ascii") if ok else None

            # Send
            self.pub.send(meta, packed)

            # Show local preview
            cv2.imshow("VDSX Producer (context)", canvas)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cam.close()
        cv2.destroyAllWindows()
        print("[vdsx] producer loop stopped")

    # ------------- Consumer loop -------------
    def run_consumer(self):
        print("[vdsx] consumer loop started")
        last = 0
        while not self.stop_flag.is_set():
            parts = self.sub.recv(block=False, timeout_ms=100)
            if not parts:
                continue
            topic, meta_json, payload = parts
            meta = json.loads(meta_json.decode("utf-8"))

            # Basic rate print
            now = time.time()
            if now - last > 1.0:
                last = now
                print(f"[rx] tokens from {meta.get('device')} | dets={meta.get('dets')} ocr={meta.get('ocr')} fps={meta.get('fps')}")

            # Optional: visualize thumbnail if present
            if meta.get("thumb_b64"):
                try:
                    jpg = base64.b64decode(meta["thumb_b64"].encode("ascii"))
                    img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
                    cv2.imshow("VDSX Consumer (thumb)", img)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                except Exception:
                    pass

        cv2.destroyAllWindows()
        print("[vdsx] consumer loop stopped")

    def stop(self):
        self.stop_flag.set()


# ============================================
# CLI
# ============================================

def build_config_from_args(args: argparse.Namespace) -> VDSXConfig:
    cfg = VDSXConfig()

    # Role
    cfg.stream.role = args.role
    cfg.stream.zmq_bind = args.bind
    cfg.stream.zmq_connect = args.connect
    cfg.stream.topic = args.topic
    cfg.stream.send_raw_thumbnails = args.send_thumbs

    # Camera
    cfg.camera.device = args.device
    cfg.camera.width = args.w
    cfg.camera.height = args.h
    cfg.camera.fps = args.fps
    cfg.camera.libcamera_gstreamer = args.pipeline

    # Detection
    cfg.detect.enabled = not args.no_detect
    cfg.detect.model = args.yolo_model
    cfg.detect.conf = args.yolo_conf
    cfg.detect.iou = args.yolo_iou
    cfg.detect.device = args.yolo_device
    if args.detect_classes:
        cfg.detect.classes = args.detect_classes

    # OCR
    cfg.ocr.enabled = not args.no_ocr
    cfg.ocr.use_paddleocr = args.ocr_paddle
    cfg.ocr.paddle_lang = args.ocr_lang
    cfg.ocr.tesseract_oem_psm = args.ocr_tess_cfg
    cfg.ocr.min_confidence = args.ocr_min_conf

    # Segmentation & Depth toggles
    cfg.segm.enabled = args.segm
    cfg.depth.enabled = args.depth

    # Tokens
    cfg.tokens.patch = args.patch
    cfg.tokens.dim = args.tok_dim
    cfg.tokens.keep_ratio = args.keep_ratio
    cfg.tokens.compress = not args.no_compress

    # Overlay
    cfg.overlay.show_boxes = not args.hide_boxes
    cfg.overlay.show_text = not args.hide_text
    cfg.overlay.show_depth = not args.hide_depth
    cfg.overlay.show_fps = not args.hide_fps

    # Context size
    cfg.context_size = (args.ctx_w, args.ctx_h)
    return cfg


def parse_args():
    ap = argparse.ArgumentParser("VDSX Updated Pipeline")

    # Role / ZMQ
    ap.add_argument("--role", choices=["producer", "consumer"], default="producer")
    ap.add_argument("--bind", default="tcp://0.0.0.0:5557", help="producer: bind addr")
    ap.add_argument("--connect", default="tcp://127.0.0.1:5557", help="consumer: connect addr")
    ap.add_argument("--topic", default="vdsx.tokens")
    ap.add_argument("--send-thumbs", action="store_true", help="send small jpeg thumbnails with tokens")

    # Camera
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--w", type=int, default=1280)
    ap.add_argument("--h", type=int, default=720)
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--pipeline", type=str, default=None,
                    help="GStreamer/libcamera pipeline string; if set, overrides device index")

    # Detection (YOLO-World)
    ap.add_argument("--no-detect", action="store_true")
    ap.add_argument("--yolo-model", default="yolov8s-world")
    ap.add_argument("--yolo-conf", type=float, default=0.30)
    ap.add_argument("--yolo-iou", type=float, default=0.45)
    ap.add_argument("--yolo-device", type=str, default="cuda:0")
    ap.add_argument("--detect-classes", nargs="+", default=None)

    # OCR
    ap.add_argument("--no-ocr", action="store_true")
    ap.add_argument("--ocr-paddle", action="store_true")
    ap.add_argument("--ocr-lang", default="en")
    ap.add_argument("--ocr-tess-cfg", default="--oem 3 --psm 6")
    ap.add_argument("--ocr-min-conf", type=float, default=0.55)

    # Segmentation/Depth toggles
    ap.add_argument("--segm", action="store_true")
    ap.add_argument("--depth", action="store_true")

    # Tokens
    ap.add_argument("--patch", type=int, default=16)
    ap.add_argument("--tok-dim", type=int, default=64)
    ap.add_argument("--keep-ratio", type=float, default=0.15)
    ap.add_argument("--no-compress", action="store_true")

    # Overlay & context
    ap.add_argument("--ctx-w", type=int, default=640)
    ap.add_argument("--ctx-h", type=int, default=360)
    ap.add_argument("--hide-boxes", action="store_true")
    ap.add_argument("--hide-text", action="store_true")
    ap.add_argument("--hide-depth", action="store_true")
    ap.add_argument("--hide-fps", action="store_true")

    return ap.parse_args()


def main():
    args = parse_args()
    cfg = build_config_from_args(args)
    app = VDSXApp(cfg)

    try:
        if cfg.stream.role == "producer":
            app.run_producer()
        else:
            app.run_consumer()
    except KeyboardInterrupt:
        pass
    finally:
        app.stop()


if __name__ == "__main__":
    main()
