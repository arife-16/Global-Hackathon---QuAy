# VQC Model Documentation

## Result
Won 1st place at QPoland Hackathon with best MSE and R² scores.

---

## What It Does

This model predicts Bitcoin closing prices using a hybrid quantum-classical neural network. The quantum circuit extracts features from 8 technical indicators, then classical layers make the final prediction.

**Architecture:** Input → Quantum Layer (3 qubits) → Dense(16) → Dense(8) → Output

---

## The Quantum Part

We use a 3-qubit variational quantum circuit. Here's how it works:

First, input features are encoded as qubit rotations (angle encoding). Then we have 3 variational layers that each do two things: apply trainable RY and RZ rotations to each qubit, then entangle neighboring qubits with CNOT gates. Finally, we measure the first qubit to get a single quantum feature.

Total quantum parameters: 18 (3 layers × 3 qubits × 2 rotation angles per qubit)

The quantum circuit is built with PennyLane and wrapped as a Keras layer so it trains end-to-end with TensorFlow.

---

## Code Walkthrough (`vqc_model.py`)

**Feature engineering functions**  
Calculate 8 technical indicators from price/volume data: RSI (5-period momentum), MACD (trend from 5/10 EMAs), ROC (3-period rate of change), ATR (normalized volatility), HL_Range (intraday spread), Volume_Change, and two lagged returns.

**Quantum circuit setup**  
Define the 3-qubit circuit with angle encoding and variational layers. Wrap it in a custom Keras layer that processes samples through the quantum circuit.

**Data loading**  
Read training data from `X_train.csv`. Calculate all 8 features using the functions above. Handle NaN values with forward then backward fill. Split into features (8 indicators) and target (returns calculated as `(Close - Open) / Open`).

**Preprocessing**  
Apply StandardScaler to normalize both features and returns to zero mean and unit variance. This helps training stability.

**Model building**  
Stack the quantum layer with two dense layers (16 and 8 units with ReLU) and a linear output. Compile with Adam optimizer and MSE loss.

**Training**  
Fit for up to 100 epochs with batch size 32. Early stopping kicks in after 15 epochs without improvement. Save the best model as `vqc_final.keras`.

**Prediction**  
Load test data from `X_test.csv`. Since it doesn't have Close prices, estimate as `(High + Low) / 2` for feature calculation. Apply the same feature engineering and scaling. Predict normalized returns, inverse transform, then convert to prices using `predicted_close = return × Open + Open`.

**Constraint handling**  
Check how many predictions fall outside the daily [Low, High] range. Print the violation count. Clip all predictions to satisfy constraints.

**Output**  
Save predictions to `predictions_vqc.csv` with Date and Close columns. Generate a visualization showing predictions against Low/High bounds, marking violations in red, plus a bar chart of constraint compliance. Save as `prediction_violations_vqc.png`.

---

## The 8 Features

1. **RSI (5)** - Momentum indicator, >70 is overbought, <30 is oversold
2. **MACD (5/10)** - Difference between fast and slow moving averages
3. **ROC (3)** - Price change rate over 3 periods  
4. **ATR (5)** - Average true range divided by Close, measures volatility
5. **HL_Range** - `(High - Low) / Open`, intraday movement
6. **Volume_Change** - Percent change in trading volume
7. **Close_lag1_pct** - Yesterday's return
8. **Close_lag2_pct** - Two-day-ago return

These capture momentum, trend, volatility, and recent behavior.

---

## Why Returns Instead of Prices?

Training data might be around $6000 while test data is around $6200. Predicting percentage returns handles this price level shift better than absolute prices. Returns are scale-invariant. After predicting the return, we convert back: `Close = return × Open + Open`.

---

## Constraint Handling

Competition rules require predictions within [Low, High]. We predict freely first, count violations (tells us model quality), then clip to enforce bounds. Lower violation rate means the model naturally learned reasonable prices.

---

## Running It

Install requirements:
```bash
pip install pandas numpy pennylane tensorflow scikit-learn matplotlib
```

Put `X_train.csv` and `X_test.csv` in the same folder as `vqc_model.py`, then run:
```bash
python vqc_model.py
```

Takes 10-15 minutes on CPU, 3-5 minutes on GPU.

---

## Outputs

- `vqc_final.keras` - trained model
- `predictions_vqc.csv` - predicted closing prices  
- `prediction_violations_vqc.png` - visualization

---

## Technical Details

**Training config:**
- Adam optimizer (default learning rate)
- MSE loss
- Batch size 32
- Max 100 epochs, early stopping patience 15
- Random seed 42

**Metrics:**
- MSE (lower better)
- R² (higher better, max 1.0)  
- Violation rate (percentage outside bounds)

**Parameters:**
- Quantum: 18
- Classical: ~300
- Total: ~318 trainable

---

## Why This Worked

The quantum circuit's entanglement can capture complex feature correlations. Predicting normalized returns handled different price regimes between train and test. The hybrid approach balances quantum advantage with practical trainability through standard backpropagation.
