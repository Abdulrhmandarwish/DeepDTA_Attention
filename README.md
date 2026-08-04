# DeepDTA Reproduction with Attention Mechanism

**Live Demo (Streamlit):** [Insert Live App URL Here]

**Drug-Target Binding Affinity Prediction — Davis & KIBA Datasets**

## Overview

This project reproduces **DeepDTA** (Öztürk, Özgür & Ozkirimli, 2018, *Bioinformatics*), a deep learning model that predicts binding affinity between a drug and a protein target directly from their sequence representations — no 3D structure required. It extends the original architecture with a custom **attention pooling layer**, adding interpretability that the original paper does not provide.

- **Paper:** [arXiv:1801.10193](https://arxiv.org/abs/1801.10193)
- **Original code/data:** https://github.com/hkmztrk/DeepDTA
- **Datasets:** Davis (68 drugs × 442 proteins, kinase inhibitors) and KIBA (2,111 drugs × 229 proteins)
- **Framework:** TensorFlow / Keras

## Why This Project

Traditional binding affinity prediction relies on molecular docking or MD simulation, which requires 3D structural data and is computationally expensive per pair. DeepDTA reframes the problem as **regression from 1D sequence data only** (SMILES string for the drug, amino acid sequence for the protein), making it scalable to thousands of drug-protein pairs where structural data doesn't exist or would be too costly to generate.

This project is Step 1 of a broader deep learning portfolio roadmap:
**DeepDTA → GraphDTA → ProteinMPNN → ESM-2 → EquiBind → TankBind → DiffDock → RFdiffusion → AlphaFold3**

## Problem Formulation

Given:
- A drug represented as a SMILES string (e.g. `CC1=C2C=C...`)
- A protein represented as an amino acid sequence (e.g. `MKKFFDSRREQ...`)

Predict:
- A continuous binding affinity value (pKd for Davis, KIBA score for KIBA) — this is a **regression** problem, not classification.

## Data Pipeline

1. **Character-level encoding.** Every character in a SMILES string or protein sequence is mapped to an integer via a fixed charset lookup table (64 possible characters for SMILES, 25 for protein sequences — the 20 standard amino acids plus a few rare/ambiguous codes).
2. **Fixed-length sequences.** Sequences are padded with zeros or truncated to a fixed length: 100 characters for SMILES, 1000 for protein sequences. This is required because neural networks need fixed-size input.
3. **Affinity matrix → training pairs.** The raw data provides an affinity matrix `Y` (drugs × proteins). This is flattened into individual `(drug, protein, affinity)` training examples, skipping any missing values (relevant for KIBA, which has genuine gaps; Davis has none).
4. **Official train/test splits.** Rather than a random split, the pre-defined fold indices bundled with the original repository are used, so results are directly comparable to the published paper.

## Model Architecture

Two parallel branches, one per input type:

**Drug (SMILES) branch:**
`Embedding (128-dim) → Conv1D(32, kernel=4) → Conv1D(64, kernel=4) → Conv1D(96, kernel=4) → Attention Pooling`

**Protein branch:**
`Embedding (128-dim) → Conv1D(32, kernel=8) → Conv1D(64, kernel=8) → Conv1D(96, kernel=8) → Attention Pooling`

(The protein branch uses a wider convolution window since protein sequences are much longer and need a wider receptive field to capture meaningful motifs.)

**Merge and prediction head:**
`Concatenate → Dense(1024) → Dropout(0.1) → Dense(1024) → Dropout(0.1) → Dense(512) → Dense(1)`

### Custom Addition: Attention Pooling

The original DeepDTA uses `GlobalMaxPooling1D` after the convolutional stack, which keeps only the single strongest activation per filter and discards *where* in the sequence that signal came from.

This reproduction replaces max pooling with a lightweight **additive attention layer**:
1. A dense layer computes an importance score for every position in the convolved sequence.
2. Scores are normalized with softmax across the sequence dimension.
3. The final representation is a weighted sum over all positions, rather than just the maximum.

This has two benefits:
- The model can combine signal from multiple informative regions instead of just one.
- The learned attention weights are directly inspectable, enabling visualization of which SMILES characters or protein residues most influenced a given prediction — a form of built-in interpretability not present in the original paper.

## Training Setup

- **Loss:** Mean Squared Error (MSE) — standard for regression
- **Optimizer:** Adam
- **Batch size:** 256
- **Epochs:** 100 (matching the paper)

## Evaluation Metrics

- **MSE** — standard regression error
- **Concordance Index (CI)** — measures whether the model correctly ranks *pairs* of samples by relative affinity, not just absolute accuracy (0.5 = random ranking, 1.0 = perfect ranking). Not available in scikit-learn; implemented manually.
- **rm²** — penalizes models that rank correctly but whose predicted values deviate from the ideal regression line. Also implemented manually, following the paper's formula.

### Reference results from the original paper

| Metric | Davis | KIBA |
|---|---|---|
| MSE | ~0.261 | ~0.194 |
| CI | ~0.878 | ~0.863 |
| rm² | ~0.630 | ~0.673 |

## Project Structure

```
.
├── notebooks/
│   └── DeepDTA_Attention_Reproduction.ipynb # Full pipeline: data → model → training → evaluation
├── app/
│   └── streamlit_app.py                     # Streamlit web application
├── models/                                  # Pre-trained Keras models (Davis & KIBA)
├── data/                                    # Sample testing data
├── README.md                                # This file
└── requirements.txt                         # Python dependencies
```

## How to Run

### 1. Web Application (Streamlit)

To run the interactive prediction app locally:

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app/streamlit_app.py
```

### 2. Notebook Pipeline

1. Open `notebooks/DeepDTA_Attention_Reproduction.ipynb` in Kaggle Notebooks or Google Colab (GPU runtime recommended).
2. Run cells top to bottom. The notebook clones the official DeepDTA data repository automatically — no manual download needed.
3. For a first pass, run with `quick_test=True` (3 epochs) to confirm the full pipeline works end-to-end before committing to the full 100-epoch run.
4. Repeat for both Davis and KIBA (both are included in the notebook).

## What Makes This More Than a Reproduction

- Faithful implementation of the original architecture, hyperparameters, and evaluation protocol, using the paper's own official data splits for a fair comparison.
- A genuine architectural extension (attention pooling in place of max pooling) that is explainable and testable, not just a superficial change.
- Visualization tooling to inspect *why* the model makes a given prediction, connecting back to prior structural biology / docking work — high-attention protein regions can be cross-referenced against known binding pocket residues.

## Next Steps in the Roadmap

This is Step 1 of a broader portfolio. Step 2, **GraphDTA**, replaces the SMILES CNN branch with a graph neural network operating directly on the molecular graph, rather than a 1D character sequence — a natural next extension building on the intuition developed here.
