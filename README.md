# Video-Depth-Anything Checkpoint Converter

This project delivers a full-stack workflow for turning Video-Depth-Anything checkpoints into ONNX models that can be executed in browsers or WebAssembly runtimes. The web UI lets you upload entire checkpoint folders, review the contents, tweak export settings, and trigger a FastAPI backend that performs the PyTorch→ONNX conversion.

## ✨ Features

- **Folder-aware uploader** – drag and drop checkpoint directories or select them via the file picker. The app keeps folder structure intact so the backend can reconstruct repositories.
- **Interactive inspection** – see file sizes, paths, and quick previews for config or text files before exporting.
- **Custom export options** – configure encoder variant, opset version, metric mode, input shapes, and dynamic axes.
- **Server-side conversion** – FastAPI endpoint recreates the model definition, loads weights, and exports an ONNX file using `torch.onnx.export`.
- **One-click download** – receive the generated ONNX file directly in the browser once conversion completes.

## 🗂 Project structure

```
.
├── index.html            # Front-end entry point
├── static/
│   ├── app.js           # Uploader logic, previews, API calls
│   └── styles.css       # Tailored styling for the experience
├── server/
│   ├── main.py          # FastAPI application exposing /api/convert
│   └── requirements.txt # Backend dependencies
└── tools/
    └── export_vda_onnx.py # CLI helper reused by the backend
```

## 🚀 Getting started

### 1. Clone repositories

```bash
git clone https://github.com/DepthAnything/Video-Depth-Anything
# Clone this project next to it
```

### 2. Backend setup

```bash
cd Token-Matrix/server
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload
```

The FastAPI server exposes the following endpoints:

- `GET /api/health` – health check returning `{ "status": "ok" }`
- `POST /api/convert` – accepts uploaded files plus form parameters and returns an ONNX file

### 3. Front-end setup

The UI is static, so any HTTP server works:

```bash
cd Token-Matrix
python -m http.server 5173
```

Open `http://localhost:5173/index.html` in your browser. The UI targets `http://<host>:8000` by default; override this by defining `window.__VDA_API_BASE__` (or `window.__VDA_API_PORT__`) before loading `static/app.js` when embedding the app elsewhere.

## 🧪 Upload expectations

1. Zip or drag the entire cloned Video-Depth-Anything repository including the `video_depth_anything` Python package.
2. Ensure the checkpoint (`*.pth`, `*.pt`, or `*.ckpt`) lives somewhere within the uploaded folder structure.
3. Choose export settings and press **Convert to ONNX**.
4. Wait for the download to start automatically.

The backend reuses the logic in `tools/export_vda_onnx.py`, so the uploaded files must provide the same environment needed for the CLI exporter to succeed.

## ⚙️ Configuration parameters

| Field | Description |
| --- | --- |
| Encoder Variant | Selects the ViT backbone (`vits`, `vitb`, or `vitl`). |
| Input Height / Width | Dummy tensor dimensions used during export. Must be multiples of 14. |
| Sequence Length | Temporal window size for the exported model. |
| Batch Size | Dummy batch size. Adjust if you plan to run multi-sample inference. |
| ONNX Opset Version | Controls the ONNX opset passed to `torch.onnx.export`. |
| Dynamic Axes | Enables dynamic batch/time/height/width axes in the ONNX graph. |
| Metric Mode | Loads the metric variant of the model head. |

## 🛡️ Error handling

- Missing checkpoints or repository code return HTTP 400 with actionable messages.
- Allocation issues or export failures return HTTP 500 with the PyTorch/ONNX error message to simplify debugging.
- Temporary upload directories are cleaned up automatically once the response is sent.

## 📄 License

This repository inherits the license terms defined in [LICENSE](LICENSE).
