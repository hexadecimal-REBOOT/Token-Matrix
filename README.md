# 🌍 VDSX Family

Visual Data Scene eXchange — Building the World in 3D, for Humans and AI

⸻

## 🔷 Overview

VDSX is a next-generation 3D capture and reasoning framework that connects vision, geometry, and semantic understanding in real time.
It turns any camera—mobile, embedded, or industrial—into a VDSX-enabled 3D scanner capable of generating structured, ontology-aware .vdsx scene files.

Each .vdsx file encodes not just images, but depth, motion, segmentation, spatial alignment, and meaning—the building blocks of a living, searchable digital twin of the real world.

The VDSX Family forms the infrastructure for planet-scale 3D understanding, where every captured frame contributes to a continuously learning world model.

⸻

## 🧩 Core Ecosystem

| Module | Purpose | Description |
| --- | --- | --- |
| VDSX | Core Capture Protocol | Defines the core .vdsx standard — integrating RGB, depth, segmentation, and ontology layers into a unified format. It establishes the foundation for real-time 3D reconstruction and AI reasoning. |
| VDSX Lite | Sensor-Enhanced Capture | Extends VDSX with IMU (Inertial Measurement Unit) and MIMO (Multi-Input Multi-Output) sensors for motion, orientation, and precision spatial tracking. Designed for mobile and field-level capture, enabling robust real-world scene reconstruction from handheld or moving cameras. |
| VDSX Hardware | Edge & Embedded Runtime | Runs the VDSX protocol directly on FPGA/SoC devices and specialized boards (e.g., Nimbus N001/N010). Supports sensor fusion inputs (IMU, LiDAR, ToF, thermal, GNSS) and deterministic 3D scene encoding at the hardware layer for ultra-low-latency applications. |
| VDSX Universal | Media, Gaming & Entertainment | The interoperability bridge connecting VDSX to creative industries. Import assets from Blender, Unreal, Unity, Maya, or GLTF, and export .vdsx data to cinematic, VR, or XR pipelines. Enables photo-realistic TSR rendering and hybrid workflows between real and synthetic scenes. |
| VDSX Stereo | Multi-View Relative Depth | Solves relative depth estimation across stereo or multi-camera configurations. Performs disparity fusion, optical alignment, and motion-based parallax reconstruction for temporally consistent 3D data. |
| VDSX Files | Data Structure & Retention | Defines the .vdsx file schema. A modular, quantum-signable container supporting retention modes (choose what to save) with file sizes from 300 KB to 4 GB. Supports depth bins, semantic layers, IMU/MIMO telemetry, and compression options for cloud or local storage. |
| VDSX Global Indexing | Planet-Scale Mapping | Distributed 3D indexing system that aggregates all .vdsx captures. Supports real-time spatial search, semantic ontology linking, and geographic query resolution. Enables a global “Search the World in 3D” experience. |
| 3DWOM | 3D World Observer Model | The AI model trained on VDSX data. Integrates geometry, semantics, and ontologies to create a reasoning model that understands the world physically — not just by pixels, but by relationships, function, and context. |

⸻

## 🧠 Why It’s Different

Today’s AI sees pixels.

Vision–language models like CLIP or GPT-4V recognize patterns, but lack geometry, permanence, and true spatial understanding.

VDSX sees the world.

VDSX integrates depth, motion, and meaning — allowing AI to understand how things exist and why they matter.

It’s not just “That’s a fire hydrant.”
It’s “That’s a fire hydrant, connected to a municipal water system, standardized in size, placed near road infrastructure, and governed by local utility regulations.”

This leap — from semantic labeling to contextual reasoning — is what makes 3DWOM the first model capable of planning and understanding in true 3D space.

⸻

## 🧱 Architecture

```
📦 VDSX Family
 ├── vdsx/                  # Core capture + reconstruction logic
 ├── vdsx-lite/             # IMU + MIMO sensor extension
 ├── vdsx-hardware/         # Edge + FPGA sensor fusion runtime
 ├── vdsx-files/            # File schema, retention, quantum signing
 ├── vdsx-universal/        # Blender/Unreal integration & exports
 ├── vdsx-stereo/           # Stereo & multi-view disparity solver
 ├── vdsx-global-indexing/  # Distributed 3D world search
 └── 3dwom/                 # 3D reasoning model training pipeline
```

Each module interoperates through the NOLYN Protocol 0.1a for secure, post-quantum AI-to-AI communication.

⸻

## 🛰️ Example Workflow

1. **Capture** — A VDSX or VDSX Lite camera captures RGB, depth, IMU, and segmentation data in real time.
2. **Generate .vdsx File** — Data is fused into a structured container with calibrated intrinsics, extrinsics, and ontology metadata.
3. **Upload & Index** — Files are uploaded to the VDSX Global Index, where they’re spatially and semantically indexed.
4. **Reconstruct & Render** — Using TSR (Triangle Splatting Rendering), 3D scenes are reconstructed with photo-realistic fidelity.
5. **Train 3DWOM** — The AI learns physical understanding from billions of .vdsx samples — enabling world-scale reasoning.
6. **Search & Query** — Users and agents can “search the world in 3D” — by object, concept, or ontology class.

⸻

## 🚀 Scaling the Real World

By deploying VDSX Lite or VDSX Hardware devices across fleets (e.g., delivery vans or municipal garbage trucks),
the system can map 80 % of the U.S. in under a year, capturing stereo and depth data at scale.

Each frame contributes to 3DWOM’s evolving understanding of Earth — creating a shared digital twin usable for urban planning, robotics, simulation, and AI training.

⸻

## 🔒 Restricted License

**VDSX Internal Research License (VIRL-1.0)**

Copyright © 2025 NOLYN / VDSX

Redistribution, modification, or derivative use is strictly prohibited without prior written consent.

Authorized for internal R&D and affiliated research projects only.

For partnerships or commercial licensing contact
[yannis@americanrobotics.io](mailto:yannis@americanrobotics.io)

⸻

## 🧭 Mission

“We’re not just scanning the world —
we’re teaching AI to understand it.”

⸻
