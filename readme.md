
### `README.md` 

````markdown
# Huggingface.co Model Downloader & Sample Inference Code Generator

**Version:** 0.1a  
**Author:** Hermann Knopp, 2023  
**Python Version:** 3.10.10 (x64)  

---

## Overview

This project allows you to:

- Download models from [Huggingface.co](https://huggingface.co) directly to your local disk.
- Automatically generate a **sample inference code** that loads the model and performs image generation.
- Display and save the generated images on your disk.

The tool uses **Diffusers and Torch libraries** and is primarily optimized for NVIDIA CUDA GPUs.

---

## System Requirements

- **Python 3.10.10** (x64)  
- **NVIDIA GPU recommended** (e.g., RTX 3060/12GB for fast inference)  
  - FP16 models for 2GB VRAM  
  - FP32 models for 4GB VRAM  
- CPU rendering is possible but much slower (see `requirements-cpu.txt` for CPU-only libraries)

---

## Installation

1. Clone or download the repository:

```bash
git clone https://github.com/username/huggingface-model-downloader.git
cd huggingface-model-downloader
````

2. Install Python libraries:

```bash
pip install -r requirements.txt
```

3. Optional: Use a virtual environment:

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

---

## Usage

1. **Select a model:**
   Visit [Huggingface.co](https://huggingface.co) and search for your desired model. Copy the model path **without the full URL**, for example:

```
CompVis/stable-diffusion-v1-4
```

2. **Run the downloader:**

```bash
python "huggingface model downloader.py"
```

3. **Enter the model path:**
   If no path is entered, the default example model `runwayml/stable-diffusion-v1-5` will be used.

4. **Download & sample code:**

   * The script sets up a dummy pipeline to download the model.
   * A ready-to-use **sample inference code** is automatically generated and saved, e.g., `Inference_Code_19112025_153200.py`.
   * This code shows how to load the model, enter prompts, and generate & save images.

5. **Image generation:**

   * Works with CPU or GPU (`pipe.to("cuda")` or `pipe.to("cpu")`)
   * Randomized seeds for varied results
   * Images are saved as `test.png` by default and displayed automatically

---

## Notes

* Models are stored in the local cache, e.g.:

```
C:\Users\<username>\.cache
```

* For slow rendering or CPU-only systems, the inference code can be adjusted (`torch.Generator("cpu")`).
* GPU-specific optimizations such as **XFormers memory-efficient attention** are included in the sample code as comments.

---

## Sample Inference Code

```python
from diffusers import StableDiffusionPipeline
import torch
import random

model_path = r"runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float32)
pipe.to("cuda")  # for CPU: pipe.to("cpu")

prompt = "a photo of an astronaut"
seed = random.randint(100,50000)
g_cuda = torch.Generator("cuda").manual_seed(seed)

image = pipe(prompt, num_inference_steps=20, guidance_scale=7.5).images[0]
image.save("test.png")
image.show()
```

---

## Contact

Email: [hermann.knopp@gmx.at](mailto:hermann.knopp@gmx.at)

---

## License

Project is private / Open Source by agreement.

```


