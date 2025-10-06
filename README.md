# Beehive-Vision
# 🐝 Baseline CNN and 2-branchs CNN+MLFB-ANN for beehive image classification

This repository provides two convolutional neural network architectures for **beehive conditions-based image classification**:

- **Baseline CNN** – a simple reference network.  
- **2-branch-CNN+MLFB-ANN** – a dual-branch CNN with a Multi-Layer Feedback ANN classifier (our proposed model).  

Both models are implemented in TensorFlow/Keras and trained on six hive condition classes:
**Ants, Healthy, QueenLoss, Robbing, SHB, Varroa.**

---

## 🧠 Repository Structure
---

## ⚙️ Setup

### Clone the repository
```bash
git clone https://github.com/SelOuaH/beehive-vision.git
cd beehive-vision

python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate

## Install dependencies
pip install -r requirements.txt
---

## 🧩 Dataset
Place the dataset inside the `data/` folder so that each subfolder represents one class:
data/
  Ants/
  Healthy/
  QueenLoss/
  Robbing/
  SHB/
  Varroa/
Each subfolder should contain its corresponding .jpg or .png images.
If your dataset is elsewhere, set the environment variable:
```bash
set IMAGE_PATH="C:\path\to\data"    # Windows
export IMAGE_PATH="/path/to/data"   # Linux/Mac

## 🚀 Run the Models
Baseline CNN
python src/baseline_cnn.py

2CNN-MLFB-ANN (Proposed Model)
python src/two_cnn_mlfb_ann.py

### Both scripts:

Load data from data/
Split into 70 % / 10 % / 20 % (train/val/test)
Save plots to figures/
Save reports to results/

## 📊 Output
| File                              | Description                   |
| --------------------------------- | ----------------------------- |
| `figures/training_curves_*.pdf`   | Accuracy/Loss curves          |
| `figures/confusion_matrix_*.pdf`  | Confusion matrix              |
| `figures/per_class_metrics_*.pdf` | Precision/Recall/F1 bar chart |
| `results/*.txt`                   | Text report with metrics      |

Example results
| Model         | Test Accuracy | Macro-F1 |
| :------------ | :-----------: | :------: |
| Baseline CNN  |     93.6 %    |   0.90   |
| 2CNN-MLFB-ANN |   **97.6 %**  | **0.96** |

## 🧾 Citation

If you use this repository, please cite it as:
@software{haddaoui_beehivevision_2025,
  author  = {Seloua Haddaoui},
  title   = {Beehive Vision: Baseline CNN and 2CNN-MLFB-ANN},
  year    = {2025},
  url     = {https://github.com/SelOuaH/beehive-vision}
}

## 📜 License

Released under the MIT License — see ...

## 📧 Contact

Seloua Haddaoui
📍 GitHub Profile
💼 LinkedIn
---
