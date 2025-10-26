# Global Hackathon — QuAy

Quantum‑Enhanced Stock Price Prediction with hybrid Variational Quantum Circuits (VQC) and classical deep learning. This repo combines practical feature engineering with a quantum layer to learn return dynamics, plus notebooks exploring Quantum‑Enhanced CNN/LSTM models and QGAF search.

## Core Aspects
- Hybrid model: a custom PennyLane‑powered VQC wrapped as a Keras layer feeding a small dense head.
- Technical features: RSI, MACD, ROC, ATR, HL range, volume change, and lagged close pct.
- Constraint awareness: predictions are clipped to `Low`/`High` bounds to avoid invalid prices.
- Artifacts: saves `vqc_final.keras`, `predictions_vqc.csv`, and `prediction_violations.png`.
- Notebooks: quantum‑enhanced CNN/LSTM variants and QGAF (quantum‑guided annealing search) exploration.

## Repository Structure
- `generate_vqc_prediction.py` — end‑to‑end VQC training, prediction, and visualization.
- `Stock_Price_Prediction_with_Quantum_Enhanced_CNN_and_QGAF.ipynb` — CNN + quantum layer.
- `Stock_Price_Prediction_with_Quantum_Enhanced_LSTM_and_QGAF.ipynb` — LSTM + quantum layer.
- `prediction_model_descirption.md` — design notes.
- `Overall_model_metric_comparison.png` — comparison summary.
- `prediction_violations.png` — produced by the VQC script.

## Requirements
- Python 3.9+ recommended.
- Packages: `pandas`, `numpy`, `pennylane`, `scikit-learn`, `tensorflow`, `matplotlib`.

Install quickly:
```
pip install pandas numpy pennylane scikit-learn tensorflow matplotlib
```
If TensorFlow install issues arise, consider a version pin compatible with your Python.

## Data Inputs
Place `X_train.csv` and `X_test.csv` in the repo root.
- Training (`X_train.csv`) must include: `Open`, `High`, `Low`, `Close`, `Volume` (optional `Date`).
- Test (`X_test.csv`) must include: `Date`, `Open`, `High`, `Low`, `Volume` (the script estimates `Close`).

## Run the VQC Model
From the repo folder:
```
python generate_vqc_prediction.py
```
Outputs:
- `vqc_final.keras` — trained hybrid model.
- `predictions_vqc.csv` — submission with `Date` and clipped `Close`.
- `prediction_violations.png` — visualization of bounds and violation summary.

## Use the Notebooks
- Install Jupyter: `pip install jupyterlab`
- Launch: `jupyter lab` and open the CNN/LSTM notebooks.
- Follow the cells to reproduce quantum‑enhanced baselines and QGAF experiments.

## Notes & Tips
- Randomness in training can cause small result variations; early stopping is enabled.
- CPU is fine; GPU accelerates TensorFlow but is not required.
- Ensure CSVs have no missing critical columns; the script forward/backward fills feature NaNs.

## License
Released under the MIT License — see `LICENSE`.
