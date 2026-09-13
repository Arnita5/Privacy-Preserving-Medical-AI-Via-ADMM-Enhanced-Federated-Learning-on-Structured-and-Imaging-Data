# Privacy-Preserving Medical AI via ADMM-Enhanced Federated Learning

A research implementation of privacy-preserving medical machine learning using **federated learning** and **ADMM-based consensus optimization** across distributed clients.

## Problem

Medical data is naturally distributed across hospitals and institutions and cannot always be centralized. This project studies collaborative model training where clients keep their local data and communicate model information instead of raw patient records.

## Approach

1. Partition training data across simulated medical clients.
2. Train a local model independently at each client.
3. Maintain local and global variables for ADMM consensus optimization.
4. Aggregate client information at the server.
5. Repeat until the global model converges.

The design can accommodate both structured/tabular and imaging pipelines.

## Architecture

```text
Client 1 ─┐
Client 2 ─┼─> Local Training ─> ADMM Consensus ─> Global Model
Client N ─┘                                      │
                    <────────────────────────────┘
```

## Repository Structure

```text
src/fedadmm.py       # Research baseline implementation
requirements.txt     # Python dependencies
README.md            # Project documentation
```

## Run

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
pip install -r requirements.txt
python src/fedadmm.py
```

## Technologies

Python, PyTorch, NumPy, scikit-learn, Federated Learning, ADMM, Medical AI.

## Privacy

No patient-identifiable information or private clinical datasets are included. Keep real medical data outside the repository.

## Note

The code in this repository is a reproducible baseline for the project methodology. Dataset-specific preprocessing, model architectures and experimental configurations can be plugged into the federated/ADMM training loop without committing sensitive data.

## Author

**Arnita N.**
