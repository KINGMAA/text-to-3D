### **Attentional Language Modeling for Text-to-3D Generation**

#### **Introduction**
Welcome to the official repository for **"Attentional Language Modeling for Text-to-3D Generation"** by **Marena Anis Labib**. This project presents a novel framework that revolutionizes 3D scene generation from textual descriptions, achieving new benchmarks in realism, consistency, and scene complexity. 

#### **Core Features**
- **Cinematographer (Trajectory Diffusion Transformer - Traj-DiT):** Generates adaptable and realistic camera trajectories using dense-view trajectory encoding, BERT embeddings, and a Multi-Head Self-Attention (MHS) mechanism.
- **Decorator (Gaussian-driven Multi-view Latent Diffusion Model - GM-LDM):** Constructs coherent scenes by aligning sparse-view pixels with 3D Gaussians.
- **Detailer (SDS++ Loss):** Enhances fidelity, ensuring high-quality, consistent multi-view outputs.

#### **Highlights**
- **State-of-the-Art Results:** 
  - BRISQUE: 23.3  
  - NIQE: 4.34  
  - CLIP-Score: 86.1
- **Applications:** Suitable for Education, simulation, virtual reality, and more.
- **Open Source:** Implementation code and resources are available for community use and development.

---

### **Paper Publication**
This paper was presented at the **ICICT 2025 Conference**. Read the full research paper here: [Access Paper](https://drive.google.com/file/d/1wTHFMfH1lDCsyivNqUjkSLnuTTBDkHIK/view)

### **Presentation**
Explore the project presentation for an in-depth overview: [View Presentation](https://docs.google.com/presentation/d/10K--cUYKWXA7ngnw8u45_wL1_V81o0VL/edit?usp=sharing&ouid=110230214992646174395&rtpof=true&sd=true)

---

### **Installation**

1. **Clone the repository:**
   ```bash
   git clone https://github.com/marinanis/text-to-3d.git
   cd text-to-3d
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up additional requirements:**
   - Ensure Python 3.8+ and a compatible CUDA environment.
   - Download datasets: [MVImgNet](#), [DL3DV-10K](#), and [LAION](#).

---

### **Usage**

#### **1. Training the Model**
To train the model on your dataset:
```bash
python train.py --config configs/config.yaml
```

#### **2. Running Inference**
Generate 3D models from text prompts:
```bash
python generate.py --input "a delicious hamburger on a wooden table"
```

#### **3. Evaluation**
Evaluate model performance using the included metrics:
```bash
python evaluate.py --model-checkpoint path/to/checkpoint
```

---

### **Results**
#### **Quantitative Comparison**
| **Method**         | **BRISQUE ↓** | **NIQE ↓** | **CLIP-Score ↑** |
|---------------------|------------------|----------------|--------------------|
| DreamFusion        | 90.2             | 10.48          | 67.4               |
| Magic3D            | 92.8             | 11.20          | 72.3               |
| LatentNeRF         | 88.6             | 9.19           | 68.1               |
| SJC                | 82.0             | 10.15          | 61.5               |
| Fantasia3D         | 69.6             | 7.65           | 66.6               |
| ProlificDreamer    | 61.5             | 7.07           | 69.4               |
| Director3D         | 32.3             | 4.35           | 85.5               |
| **Ours**           | **23.3**         | **4.34**       | **86.1**           |

#### **Qualitative Results**
[Watch Demo](https://drive.google.com/file/d/13g-qvsW4usYLyVQDr-dm7prLUZjF8UcN/view?usp=sharing)

---

### **Contributing**
We welcome contributions! Please check our [Contributing Guidelines](CONTRIBUTING.md) for details on how to get involved.

---

### **Citation**
If you use this code or find our work helpful, please cite:
```bibtex
@article{anis2025textto3d,
  title={Attentional Language Modeling for Text-to-3D Generation},
  author={Marena Anis Labib and Ali Hamdi},
  journal={ICICT 2025 Conference Proceedings},
  year={2025}
}
```

---

### **Acknowledgments**
- Inspired by work of [Director3D](https://arxiv.org/pdf/2406.17601v1).
- Utilizes datasets: MVImgNet, DL3DV-10K, and LAION.
- The model employs **BERT embeddings** to encode textual descriptions into semantic representations for accurate scene generation.

---

### **License**
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more information.

