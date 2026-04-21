# 🧠 NeuroVoice PD — Parkinson's Disease Detection & Severity Estimation via Voice Analysis

> A multi-output deep learning system that detects Parkinson's Disease and estimates symptom severity from biomedical voice measurements.

---

## 📌 Overview

**NeuroVoice PD** is a dual-task neural network that simultaneously:
- **Classifies** whether a patient has Parkinson's Disease (binary: Healthy vs. Parkinson's)
- **Estimates** the severity of symptoms on a 0–4 scale using voice biomarkers

The model is built using TensorFlow/Keras and trained on the [UCI Parkinson's Dataset](https://archive.ics.uci.edu/ml/datasets/parkinsons), which contains biomedical voice measurements from 31 individuals (23 with Parkinson's Disease).

---

## 🗂️ Project Structure

```
neurovoice-pd/
│
├── parkinsons_model.py       # Main script (data loading, model, training, evaluation)
├── parkinsons.data           # Dataset (UCI Parkinson's Dataset)
├── parkinson_voice_model.h5  # Saved trained model (generated after training)
├── requirements.txt          # Python dependencies
└── README.md
```

---

## 🧬 Dataset

The dataset contains **195 voice recordings** from 31 people. Each row is one voice recording, with 22 biomedical voice features as inputs.

| Column | Description |
|--------|-------------|
| `MDVP:Fo(Hz)` | Average vocal fundamental frequency |
| `MDVP:Jitter(%)` | Measure of variation in fundamental frequency |
| `MDVP:Shimmer` | Measure of variation in amplitude |
| `NHR`, `HNR` | Noise-to-harmonics ratios |
| `RPDE`, `D2` | Nonlinear dynamical complexity measures |
| `DFA` | Signal fractal scaling exponent |
| `PPE` | Nonlinear measure of fundamental frequency variation |
| `status` | **Target** — 1 = Parkinson's, 0 = Healthy |

> Download the dataset from: https://archive.ics.uci.edu/ml/datasets/parkinsons  
> Place it in the project root as `parkinsons.data`.

---

## 🏗️ Model Architecture

The model uses a **shared encoder + dual-branch** design:

```
Input (22 features)
      │
 [Dense 128 → BN → Dropout 0.4]
 [Dense  64 → BN → Dropout 0.3]
      │
  ┌───┴───┐
  │       │
Classification    Regression
Branch            Branch
[Dense 32]        [Dense 32]
[Dropout 0.2]     [Dropout 0.2]
[Sigmoid]         [Linear]
  │                   │
Binary Output     Severity Score
(0 = Healthy,     (0–4 scale)
 1 = Parkinson's)
```

**Loss function:** Weighted combination — `0.6 × Binary Crossentropy + 0.4 × MSE`

---

## ⚙️ Installation

```bash
git clone https://github.com/MariamSamir99/neurovoice-pd.git
cd neurovoice-pd
pip install -r requirements.txt
```

### `requirements.txt`

```
numpy
pandas
scikit-learn
tensorflow>=2.10
matplotlib
seaborn
```

---

## 🚀 Usage

1. Place the dataset file `parkinsons.data` in the project root (or update the path in the script).

2. Run the training script:

```bash
python parkinsons_model.py
```

This will:
- Load and preprocess the data
- Split into train (70%) / validation (15%) / test (15%)
- Train the dual-output model for up to 200 epochs with early stopping
- Print classification report and evaluation metrics
- Display training curves, confusion matrix, and severity scatter plot
- Save the trained model as `parkinson_voice_model.h5`

---

## 📊 Evaluation Metrics

The model is evaluated on three metrics:

| Metric | Task |
|--------|------|
| **Accuracy** | Binary classification (Healthy / Parkinson's) |
| **AUC** | Binary classification (area under ROC curve) |
| **MAE** | Severity regression (mean absolute error on 0–4 scale) |

---

## 📈 Outputs & Visualizations

After training, the script generates:

- **Confusion Matrix** — classification performance on the test set
- **Severity Scatter Plot** — predicted vs. actual severity scores
- **Training History** — accuracy and MAE curves over epochs

---

## 🧪 Data Split

| Split | Size |
|-------|------|
| Training | 70% |
| Validation | 15% |
| Test | 15% |

Stratified splitting is used to preserve class balance across all splits.

---

## 🛡️ Training Details

| Hyperparameter | Value |
|----------------|-------|
| Optimizer | Adam (lr = 0.0005) |
| Epochs | Up to 200 (early stopping) |
| Batch Size | 16 |
| Early Stopping Patience | 20 epochs |
| LR Reduction Patience | 10 epochs (factor 0.5) |
| Weight Initialization | He Normal |

---

## ⚠️ Disclaimer

This project is for **educational and research purposes only**. It is not intended for clinical diagnosis or medical use. Always consult a licensed medical professional for health-related decisions.

---

## 📄 License

This project is open-source under the [MIT License](LICENSE).

---

## 🙏 Acknowledgements

- Dataset: [UCI Machine Learning Repository — Parkinson's Dataset](https://archive.ics.uci.edu/ml/datasets/parkinsons)  
- Original study: Max A. Little, University of Oxford
