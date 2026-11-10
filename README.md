# CRSD: Context–Residual Spectral Detector

Official implementation of **CRSD**, a label-efficient graph anomaly detector.

## Requirements

```bash
pip install -r requirements.txt
```

A CUDA-capable GPU is recommended.

## Datasets

Create a `dataset` folder in the project root and put the datasets in it:

```
CRSD/
├── dataset/
│   ├── elliptic
│   └── tfinance
├── data.py
├── models.py
├── run.py
└── requirements.txt
```

- **YelpChi** and **Amazon**: loaded through DGL (`FraudYelpDataset` / `FraudAmazonDataset`) and stored in `dataset/`.

On the first run, the processed data are cached in `dataset/processed/`.

## Usage

Run CRSD on all four datasets with the default seeds (42, 52, 62):

```bash
python run.py
```

Run on selected datasets or seeds:

```bash
python run.py --datasets yelp amazon --seeds 42 52 62
```

| Argument | Default | Description |
|---|---|---|
| `--datasets` | `yelp amazon elliptic tfinance` | Datasets to evaluate |
| `--seeds` | `42 52 62` | Random seeds for data splits and training |
| `--data-root` | `./dataset` | Dataset folder |
| `--device` | `cuda` | Device for the spectral backbone |
| `--output` | `results.json` | File for the results |

Each run samples 20 anomalies and 80 normal nodes for training and a disjoint validation set of the same size; all remaining labeled nodes are used for testing. Macro-F1 and AUC for each seed are printed and saved to `results.json`.
