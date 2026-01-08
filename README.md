# SonarSAM 2.1: Segment Anything for Marine Debris Detection

**SonarSAM 2.1** is a specialized adaptation of Meta's **Segment Anything Model 2 (SAM 2.1)** for Forward-Looking Sonar (FLS) imagery. It is designed to robustly detect and segment marine debris (bottles, tires, propellers, etc.) in challenging underwater acoustic environments.

## 🚀 Key Features

Unlike standard optical segmentation models, SonarSAM 2.1 addresses specific sonar challenges:

* **⚡ Backbone Upgrade:** Migrated from ViT (SAM 1) to **Hiera-Tiny (SAM 2.1)** for hierarchical multi-scale feature extraction and faster inference.
* **🌊 CLAHE Preprocessing:** Implements **Contrast Limited Adaptive Histogram Equalization** to suppress speckle noise and enhance debris acoustic shadows.
* **🎯 Box Jitter Robustness:** Randomly jitters bounding box prompts (±20px) during training to force the model to learn object shapes rather than relying on perfect prompts.
* **⚖️ Weighted Focal Loss:** Custom loss function that weights debris classes **20x higher** than background to handle extreme class imbalance (99% water vs 1% object).
* **🔄 Consistency Regularization:** (Optional) Enforces prediction invariance between original and horizontally flipped sonar images.

---

## 🛠️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/SonarSAM-2.1.git
cd SonarSAM-2.1
```

### 2. Set up Environment

We recommend using Conda to manage dependencies.

```bash
conda create -n sonarsam python=3.10 -y
conda activate sonarsam

# Install PyTorch (Adjust CUDA version as needed)
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia

# Install dependencies
pip install -r requirements.txt
```

### 3. Download Checkpoints

Download the official SAM 2.1 Hiera-Tiny weights:

```bash
cd checkpoints
wget https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2.1_hiera_tiny.pt
cd ..
```

---

## 📂 Data Preparation

The project expects the **Marine Debris Datasets (Watertank)** format.
Structure your data as follows:

```
dataset/
├── marine_debris/
│   ├── train.txt  (List of training image filenames)
│   ├── test.txt   (List of testing image filenames)
└── md_fls_dataset/
    └── data/
        └── watertank-segmentation/
            ├── Images/    (*.png)
            ├── Masks/     (*.png)
            └── Box_XMLs/  (*.xml)
```

---

## 🚂 Training

To train the model from scratch (fine-tuning SAM 2.1 on your data):

```bash
python train_SAM.py   --config ./configs/sam.yaml   --save_path ./saves/run_experiment_1
```

**Key Arguments:**

* `--config`: Path to the YAML configuration file.
* `--save_path`: Directory to save checkpoints and logs.

**Output:**

* Best model saved as: `./saves/run_experiment_1/SonarSAM2_best.pth`
* Training logs: `./saves/run_experiment_1/YYYYMMDD.log`

---

## 📊 Evaluation & Visualization

## 📈 Performance

| Class | Mobile | Base SAM 2.1 | **SonarSAM 2.1** | Improvement |
| --- | --- | --- | --- | --- |
| **Bottle** | 0.767 | 0.808 | **0.823** | +7.31% |
| **Propeller** | 0.670 | 0.736 | **0.761** | +13.56% |
| **Valve** | 0.599 | 0.671 | **0.697** | +16.41% |
| **Mean IoU** | 0.704 | 0.747 | **0.770** | **+9.93%** |

---

## 📜 Citation

If you use this code, please cite the original SonarSAM paper and the SAM 2 project:

```bibtex
@article{wang2023sonarsam,
  title={When SAM Meets Sonar Images},
  author={Wang, Lin et al.},
  journal={arXiv preprint arXiv:2306.14109},
  year={2023}
}

@article{ravi2024sam2,
  title={SAM 2: Segment Anything in Images and Videos},
  author={Ravi, Nikhila et al.},
  journal={arXiv preprint arXiv:2408.00714},
  year={2024}
}
```

## 📄 License

Licensed under the Apache 2.0 License.
