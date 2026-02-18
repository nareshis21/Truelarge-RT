# Python Utilities & Scripts

This directory contains Python scripts used for maintaining the repository, optimizing assets, and converting models.

## 🛠️ Repository Utilities
These scripts are located in the `/scripts` folder.

### 1. `optimize_images.py`
Automates the optimization and formatting of documentation images.
*   **Location**: `scripts/optimize_images.py`
*   **Dependency**: `Pillow` (`pip install Pillow`)
*   **Features**:
    *   **Circular Crop**: Automatically detects the app icon and crops it to a circle.
    *   **Compression**: Resizes/re-saves all images in the `docs/` folder to ensure they are under 1MB while maintaining readability.

**Usage**:
```bash
python scripts/optimize_images.py
```

---

## 🧠 Model Conversion Scripts
These scripts are part of the `llama.cpp` core and are used to prepare GGUF models for use in the app.

### 1. `convert_hf_to_gguf.py`
The primary script for converting HuggingFace models (Safetensors/PyTorch) to the GGUF format used by TrueLarge-RT.
*   **Location**: `app/src/main/cpp/llama.cpp/convert_hf_to_gguf.py`

### 2. `convert_lora_to_gguf.py`
Used for converting LoRA adapters for on-device inference.
*   **Location**: `app/src/main/cpp/llama.cpp/convert_lora_to_gguf.py`

### 3. `json_schema_to_grammar.py`
Converts JSON schemas into GBNF grammars, enabling structured output (JSON mode) in the Android app.
*   **Location**: `app/src/main/cpp/llama.cpp/examples/json_schema_to_grammar.py`

---

## ⚙️ Environment Setup
To use these scripts, ensure you have Python 3.10+ installed and install the required dependencies:

```bash
pip install -r app/src/main/cpp/llama.cpp/requirements.txt
pip install Pillow
```
