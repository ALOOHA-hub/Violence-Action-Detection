# 🚨 Violence Action Detection

**Violence Action Detection** is a multi-modal, asynchronous Vision-Language pipeline designed for privacy-preserving. It detects violent physical altercations and automatically generates detailed forensic JSON reports.
---


---

## Overview & Example Output

When processing an input video, the system runs. If an altercation is detected, it flags the subjects, locks their bounding boxes, and triggers a Vision-Language Model (VLM) to generate a detailed forensic JSON report.

![Video](./data/outputs/incident_20260223-1641271.mp4)

```json
{
    "threat_detected": true,
    "description": "Two individuals are engaged in a physical altercation. One person is wearing a red hoodie and camouflage pants, while the other is dressed in a white shirt and blue jeans. The person in the red hoodie appears to be pushing or shoving the other individual, who is trying to maintain balance and distance. The scene is set in an outdoor area with a house and parked cars in the background."
}
```

---

## ✨ Key Engineering Features

- **Asynchronous Pipeline**: Utilizes a Producer-Consumer threading model to decouple high-speed camera rendering from heavy AI inference, ensuring zero UI blocking during complex analysis.
- **"Infinite Latch" 3-State Memory**: A robust temporal memory layer (`SecurityStateManager`) tracks individuals across three states:
  - 🟢 **Green (Safe)**: Normal behavior detected.
  - 🟠 **Orange (Suspicious)**: 3-Strike escalation based on action recognition. Triggers event recording.
resetting their threat level.
- **VLM Strategy Pattern**: The Vision-Language Model layer is decoupled via a Factory and Strategy pattern (`VisionReasonerFactory`), allowing seamless hot-swapping between:
  - Local Edge Models (`qwen2.5vl:3b` via Ollama).
  - Cloud API Models (`qwen-vl-max` via DashScope).
- **Zero-Shot Action Recognition**: Utilizes contrastive text-image pre-training (CoCa ViT-L-14) to dynamically classify behavior from raw text prompts without needing a custom-trained action dataset.

---

## 🧠 System Architecture (The 3 Phases)

The pipeline is split into three phases:

### 1. Phase 1: Spatial Perception (The Eyes)
* **Engines**: `YOLOv8` + `ByteTrack` (via `supervision`)
* **Role**: Runs on every frame. Detects humans (COCO Class 0), assigns persistent temporal Tracker IDs, and dynamically updates bounding boxes.

### 2. Phase 2: Temporal Action Recognition (The Reflexes)
* **Engines**: `OpenCLIP (CoCa ViT-L-14)`
* **Role**: Extracts 8-frame micro-clips (stride 4) of specific tracked IDs asynchronously. Evaluates behavior against predefined text prompts (e.g., "a person punching", "normal behavior"). If violent actions exceed a threshold, it triggers a "strike". 3 strikes escalate the subject to **Orange** and begins incident recording.

### 3. Phase 3: Forensic Reasoning (The Brain)
* **Engines**: `Qwen2.5-VL` (Local) or `Qwen-VL-Max` (Cloud API)
* **Role**: Triggered after an incident recording finishes. Extracts 8 keyframes from the recorded footage and passes them to the VLM. It assesses spatial relationships, identifies context, and outputs a structured **JSON Forensic Report**.

---

## 📁 Directory Structure

```text
 Violence Action Detection/
 ├── configs/
 │   └── config.yaml              # Core configurations for all 3 phases
 ├── data/
 │   ├── inputs/                  # Source CCTV videos (.mp4)
 │   └── outputs/                 # Auto-generated incident recordings and JSON reports
 ├── models/                      # Downloaded weights (YOLO, CoCa .bin files)
 ├── src/
 │   ├── core/
 │   │   ├── analysis/            # Phase 2 (Action Recognizer) & Phase 3 (VLM Factory)
 │   │   ├── memory/              # SecurityStateManager & Evidence buffers
 │   │   └── perception/          # Phase 1 YOLOv8 Detector
 │   ├── pipelines/               # Rapid pipeline multi-threading logic
 │   ├── utils/                   # Config loader, logger, visualization helpers
 │   └── main.py                  # Pipeline entry-point
 ├── tests/                       # Unit and integration tests
 ├── requirements.txt
 └── README.md
```

---

## 🛠️ Installation & Setup

### 1. Environment Requirements
- An NVIDIA GPU with CUDA support
- [Ollama](https://ollama.com/) (Required for Local VLM)

### 2. Set the env & Install the requirements
```bash
# 1. Setup Virtual Environment
python -m venv venv
source venv/bin/activate
# Windows: venv\Scripts\activate

# 2. Install Python Dependencies
pip install -r requirements.txt
```

### 3. Download Models
- Ensure your YOLOv8 weights are placed in `models/`.
- Download the CoCa ViT-L-14 (`coca_l14.bin`) weights and place them in the `models/` directory.

> **Local VLM Setup (Phase 3)**:
> Pull the Qwen Vision model via Ollama:
> ```bash
> ollama pull qwen2.5vl:3b
> ```

---


## Running the Pipeline
```bash
python -m src.main
```
