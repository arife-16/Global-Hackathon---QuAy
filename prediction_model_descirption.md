# VQC Stock Price Prediction - Submission

## Technical Specifications
- **Language**: Python 3.8+
- **Quantum Framework**: PennyLane 0.30+
- **ML Framework**: TensorFlow 2.x / Keras
- **Required Packages**:
```bash
  pip install pandas numpy pennylane tensorflow scikit-learn matplotlib
```

## Model Architecture
- **Quantum Layer**: 3-qubit Variational Quantum Circuit (VQC)
  - Angle encoding (RY gates) for 8 input features
  - 3 variational layers with RY/RZ rotations (18 trainable parameters)
  - CNOT entanglement between adjacent qubits
  - PauliZ expectation measurement
- **Classical Layers**: Quantum output → Dense(16, ReLU) → Dense(8, ReLU) → Dense(1)
- **Input Features**: RSI, MACD, ROC, ATR, HL_Range, Volume_Change, Close_lag1_pct, Close_lag2_pct

## Training Process
1. Feature engineering from OHLCV data
2. StandardScaler normalization for features and target
3. Target: Normalized returns `(Close - Open) / Open`
4. Optimizer: Adam, Loss: MSE
5. Training: 100 epochs, batch size 32, early stopping (patience=15)
6. Inverse transform predictions to price scale
7. Clip to [Low, High] constraints

## Model Comparison (QPoland Strategy)
Evaluated quantum approaches:
- **Linear Regression**: Classical baseline
- **QSVM**: Quantum kernel SVM
- **QLSTM**: Quantum-LSTM hybrid
- **QGAN**: Quantum GAN
- **VQC** (selected): Best balance of quantum advantage, training stability, and low violation rate

## Reproduction Steps
1. Install dependencies: `pip install pandas numpy pennylane tensorflow scikit-learn matplotlib`
2. Place `X_train.csv` and `X_test.csv` in working directory
3. Run: `python vqc_model.py`
4. Outputs: `predictions_vqc.csv`, `prediction_violations.png`, `vqc_final.keras`

## Validation
Violation analysis included to monitor constraint adherence before clipping.
