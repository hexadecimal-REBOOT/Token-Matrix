# Token-Matrix Documentation

This directory contains technical documentation, vision papers, and implementation guides for the VDSX (Vision-Depth-Streaming-eXtraction) system.

## Documents

### Technical Papers

- **[VDSX_OCR_Technical_Paper.md](VDSX_OCR_Technical_Paper.md)** - Complete technical vision paper describing the VDSX OCR system architecture, including vision-token streaming, OCR integration, and vehicle perception capabilities. This represents the full roadmap and design goals for the platform.

### Implementation Status

- **[Implementation_Status.md](Implementation_Status.md)** - Current implementation status mapping the vision paper components to actual codebase features.

## Quick Overview

**VDSX OCR** is a vision-token streaming system designed for efficient vehicle perception. Instead of streaming raw video, the system:

1. Processes multi-camera feeds through depth estimation and object detection
2. Encodes visual information into compact tokens
3. Integrates OCR for text extraction from the environment
4. Streams lightweight token packets for analysis and storage

This dramatically reduces bandwidth (~10× reduction) while maintaining rich semantic information for safety, analytics, and reasoning tasks.

## Current Focus

The Token-Matrix repository currently implements:
- Web-based depth model conversion (Video-Depth-Anything → ONNX)
- Real-time camera capture and depth processing
- Point cloud generation from depth maps
- VDSX multi-frame processing interface

See [Implementation_Status.md](Implementation_Status.md) for detailed component mapping.
