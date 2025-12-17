
# Extended Digit Recognition on MNIST  
## Comparative Study: Architecture, Learning Paradigm, and Training Protocol

This project is a comparative study on **handwritten digit recognition** using the **MNIST** dataset.  
We implement and evaluate three approaches:

- **MLP (Multilayer Perceptron)** — fully connected network on flattened images  
- **CNN (Convolutional Neural Network)** — convolution + pooling to exploit image structure  
- **Autoencoder → Classifier (AE→Classifier)** — unsupervised representation learning + supervised classification

The study is done by a team of **3 people**, where each person focuses on one major comparison axis:

- **Person 1 (Architecture & Inductive Bias):** how the model structure affects learning and generalization  
- **Person 2 (Learning Paradigm & Data Efficiency):** supervised learning vs unsupervised transfer (representation reuse)  
- **Person 3 (Optimization, Training Protocol & Evaluation):** how training settings affect results and behavior  

---

## Goals of the Project

1. Compare performance of MLP, CNN, and AE→Classifier on MNIST.
2. Understand **why** CNN usually performs better on images (inductive bias).
3. Check if an autoencoder learns a meaningful latent representation without labels.
4. Analyze training curves and error patterns (confusion matrix, misclassified examples).

---

## Dataset

We use the official **MNIST** dataset:

- 60,000 training images  
- 10,000 test images  
- 28×28 grayscale images  
- 10 classes (digits 0–9)

---

## Main Results (Example)

- **CNN** achieves the highest test accuracy (best generalization on images).  
- **MLP** performs well but is weaker because it does not use spatial structure.  
- **AE→Classifier** is efficient in parameters but usually below CNN because reconstruction features are not always the most discriminative.

(Your final numbers may vary depending on seeds and training settings.)

---

## Evaluation Protocol (Fair Testing)

To keep evaluation strict and fair, we follow this protocol:

- Use the **official 10,000-image test set** only for final testing  
- Evaluate using `model.eval()` and `torch.no_grad()`  
- **No test-time augmentation**

---

## Repository Contents

Typical files you may find in this repo:

- `*.ipynb` notebook(s): training + plots + visualizations  
- `report/` or `paper/`: LaTeX report and PDF  
- `figures/`: plots (learning curves, reconstructions, confusion matrices, etc.)  
- `models/` or `src/`: model definitions and training code (if included)

---

## How to Run (Basic)

### 1) Install requirements
```bash
pip install torch torchvision matplotlib numpy scikit-learn
=======
# Extended Digit Recognition - Comparative Study

Comparative MNIST study exploring four setups inside a single notebook: a fully-connected MLP baseline, a small CNN, an unsupervised autoencoder, and a classifier initialized from the trained AE encoder. Everything lives in `Extended_Digit_Recognition_–_Comparative_Study_(MLP_·_CNN_·_AE_·_AE→Classifier.ipynb`.

## Contents
- Notebook with experiments, evaluation, and commentary.
- `fig_ae_latent_pca_2d.png`: PCA visualization of the autoencoder latent space.
- `models/summary_single_models.csv`: compact metrics table retained for reference.
- `data/` (MNIST download cache) and model checkpoints are gitignored to keep the repo light.

## Setup
1) Create an environment (example with venv):
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
2) Install dependencies (adjust the PyTorch command per your platform if needed):
   ```bash
   pip install torch torchvision torchaudio numpy pandas scikit-learn matplotlib jupyter
   ```

## Run the notebook
1) Launch Jupyter:
   ```bash
   jupyter lab  # or: jupyter notebook
   ```
2) Open the notebook file and run cells in order. The MNIST dataset will download automatically into `data/` on first run.
3) Trained weights are saved under `models/` (ignored by git). Keep any metrics you want to track in text/CSV form.

## Results snapshot
Strict test accuracy for the single-model runs:

| Model           | Accuracy | Params   |
| --------------- | -------- | -------- |
| MLP             | 0.9782   | 535,818  |
| CNN (single)    | 0.9944   | 1,701,578|
| AE -> Classifier| 0.9745   | 476,490  |

Full details and error analysis are in the notebook sections 5–10.

## Publish to GitHub
1) Initialize the repo and commit tracked files:
   ```bash
   git init
   git add .
   git commit -m "Add digit recognition study notebook"
   ```
2) Create a GitHub repo, then connect and push:
   ```bash
   git branch -M main
   git remote add origin git@github.com:<your-username>/<repo-name>.git
   git push -u origin main
   ```
3) Verify `data/` and model checkpoints stay untracked (`git status` should show a clean tree).
>>>>>>> d137716 (Add MNIST comparative study notebook)
