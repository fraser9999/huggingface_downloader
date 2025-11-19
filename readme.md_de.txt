
### `README.md`

````markdown
# Huggingface.co Model Downloader & Sample Inference Code Generator

**Version:** 0.1a  
**Author:** Hermann Knopp, 2023  
**Python Version:** 3.10.10 (x64)  

---

## Übersicht

Dieses Projekt ermöglicht:

- Herunterladen von Modellen von [Huggingface.co](https://huggingface.co) direkt auf die Festplatte.
- Automatische Erstellung eines **Sample-Inferenzcodes**, der das Modell direkt lädt und für Bildgenerierung verwendet.
- Anzeige und Speicherung der erzeugten Bilder auf der Festplatte.

Das Tool verwendet die **Diffusers- und Torch-Bibliotheken** und ist primär für NVIDIA CUDA-Grafikkarten optimiert.

---

## Systemanforderungen

- **Python 3.10.10** (x64)  
- **NVIDIA GPU empfohlen** (z.B. RTX 3060/12GB für schnelle Inferenz)  
  - FP16 Modelle für 2GB VRAM  
  - FP32 Modelle für 4GB VRAM  
- CPU-Rendering möglich, aber deutlich langsamer (siehe `requirements-cpu.txt` für CPU-only Libraries)

---

## Installation

1. Repository klonen oder herunterladen:

```bash
git clone https://github.com/username/huggingface-model-downloader.git
cd huggingface-model-downloader
````

2. Python Libraries installieren:

```bash
pip install -r requirements.txt
```

3. Optional: Virtuelle Umgebung verwenden:

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

---

## Nutzung

1. **Modell auswählen:**
   Besuche [Huggingface.co](https://huggingface.co) und suche dein gewünschtes Modell. Kopiere den Modellpfad ohne die vollständige URL, z.B.:

```
CompVis/stable-diffusion-v1-4
```

2. **Downloader starten:**

```bash
python "huggingface model downloader.py"
```

3. **Modellpfad eingeben:**
   Wenn kein Pfad eingegeben wird, wird automatisch das Beispielmodell `runwayml/stable-diffusion-v1-5` genutzt.

4. **Download & Sample-Code:**

   * Das Skript erstellt einen Dummy-Pipeline-Aufruf zum Herunterladen des Modells.
   * Ein fertiger **Sample-Inferenzcode** wird automatisch generiert und gespeichert, z.B. `Inference_Code_19112025_153200.py`.
   * Der Code zeigt, wie das Modell geladen wird, Prompts eingegeben werden und Bilder generiert & gespeichert werden.

5. **Bildgenerierung:**

   * CPU oder GPU nutzbar (`pipe.to("cuda")` oder `pipe.to("cpu")`)
   * Randomisierte Seeds für unterschiedliche Ergebnisse
   * Standardmäßig werden die Bilder als `test.png` gespeichert und angezeigt.

---

## Hinweise

* Modelle werden im lokalen Cache gespeichert, z.B.:

```
C:\Users\<username>\.cache
```

* Für langsames Rendering oder CPU-Only Systeme kann der Inferenzcode angepasst werden (`torch.Generator("cpu")`).
* GPU-spezifische Optimierungen wie **XFormers Memory Efficient Attention** sind bereits im Sample-Code kommentiert.

---

## Beispiel Inferenzcode

```python
from diffusers import StableDiffusionPipeline
import torch
import random

model_path = r"runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float32)
pipe.to("cuda")  # für CPU: pipe.to("cpu")

prompt = "a photo of an astronaut"
seed = random.randint(100,50000)
g_cuda = torch.Generator("cuda").manual_seed(seed)

image = pipe(prompt, num_inference_steps=20, guidance_scale=7.5).images[0]
image.save("test.png")
image.show()
```

---

## Kontakt

Email: [hermann.knopp@gmx.at](mailto:hermann.knopp@gmx.at)

---

## Lizenz

Projekt ist privat / Open Source nach Absprache.

````

---

