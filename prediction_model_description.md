# Model Descriptions

## Competition Result
Won 1st place at QPoland Hackathon with the best MSE and R² scores across all teams.

---

## Overview

We tested three different quantum-hybrid architectures for Bitcoin price prediction. Each model uses the same 8 technical indicators but applies different quantum techniques to learn patterns in the data.

All three models follow a similar approach:
- Extract features from price/volume data (RSI, MACD, ROC, ATR, and others)
- Predict percentage returns instead of absolute prices
- Apply standard scaling to normalize inputs
- Enforce price constraints by clipping predictions to the daily [Low, High] range

---

## Model 1: Variational Quantum Circuit (VQC)

The VQC model uses a parameterized quantum circuit as the first layer, feeding its output into classical neural network layers.

### How it works

The quantum circuit has 3 qubits. First, we encode our 8 input features using angle encoding (rotating qubits based on feature values). Then the circuit has 3 variational layers where each layer applies trainable rotations (RY and RZ gates) followed by CNOT gates to entangle neighboring qubits. We measure the first qubit to get a single quantum feature, which flows into classical dense layers for the final prediction.

This gives us 18 trainable quantum parameters (3 layers × 3 qubits × 2 rotation angles) plus the weights in the classical layers.

### Notebook structure (`1_VQC_model.ipynb`)

The notebook walks through:

1. **Setup** - Import libraries (PennyLane, TensorFlow, scikit-learn) and set random seeds
2. **Feature functions** - Functions to calculate RSI, MACD, ROC, and ATR from price data
3. **Quantum circuit** - Define the 3-qubit circuit with PennyLane and wrap it as a Keras layer
4. **Load training data** - Read the CSV, compute all 8 features, handle missing values
5. **Preprocessing** - Scale features and target returns to zero mean, unit variance
6. **Build model** - Stack quantum layer → Dense(16) → Dense(8) → Dense(1)
7. **Training** - Fit for up to 100 epochs with early stopping (patience=15)
8. **Test predictions** - Load test data, predict returns, convert to prices
9. **Constraint checking** - Count how many predictions fell outside [Low, High] bounds
10. **Save and visualize** - Output CSV file and create plots showing predictions vs constraints

### Outputs
- `vqc_final.keras` - saved model weights
- `predictions_vqc.csv` - predicted closing prices
- `prediction_violations_vqc.png` - visualization of constraint violations

---

## Model 2: CNN + Quantum Gramian Angular Field

This model transforms time series into images using a quantum-inspired technique, then applies convolutional neural networks to recognize patterns.

### The QGAF transformation

Gramian Angular Field (GAF) is a way to convert a 1D time series into a 2D image while preserving temporal relationships. Here's the process:

1. Take a window of time series values and normalize them to [-1, 1]
2. Map each value to an angle: φ = arccos(value)
3. Create a matrix where each element GAF[i,j] represents the relationship between time points i and j
4. For GASF (summation): GAF[i,j] = cos(φ_i + φ_j)
5. For GADF (difference): GAF[i,j] = sin(φ_i - φ_j)

The result is an image where pixel intensities encode how different time points relate to each other in the angular domain. This quantum-inspired encoding captures phase relationships that might be missed by standard approaches.

### Notebook structure (`2_CNN_QGAF_model.ipynb`)

1. **Setup and features** - Same imports and feature engineering as the VQC model
2. **QGAF functions** - Implement the Gramian Angular Field transformation
3. **Create image dataset** - Apply QGAF to sliding windows of the time series (typically 20-30 time steps per window)
4. **CNN architecture** - Stack Conv2D layers (32 and 64 filters) with MaxPooling, then flatten and add dense layers
5. **Training** - Train on QGAF images for 50-100 epochs with early stopping
6. **Test predictions** - Transform test data to QGAF images and predict
7. **Constraint handling** - Check violations and clip to [Low, High]
8. **Results** - Save predictions and create visualizations

### Outputs
- `predictions_cnn.csv`
- `results_cnn.png`

---

## Model 3: LSTM + Quantum Gramian Angular Field

Uses the same QGAF transformation as the CNN model, but feeds the quantum-encoded features into an LSTM network instead of convolutional layers.

### Why LSTM with QGAF?

LSTMs excel at learning long-term patterns in sequential data through their gated memory cells. By preprocessing the time series with QGAF, we give the LSTM quantum-enhanced features that encode phase relationships between time points. The LSTM then learns which temporal dependencies matter for prediction.

Unlike the CNN which treats the QGAF as a spatial image, we flatten the QGAF matrix into a sequence that the LSTM processes step-by-step.

### Notebook structure (`3_LSTM_QGAF_model.ipynb`)

1. **Setup and features** - Standard imports and feature calculations
2. **QGAF transformation** - Same as CNN model but reshape output for sequential processing
3. **Sequence preparation** - Create windows, apply QGAF, flatten to 1D sequences
4. **LSTM architecture** - LSTM layer(s) with 64-128 units, dropout for regularization, dense output layers
5. **Training** - Fit model with early stopping
6. **Predictions** - Process test sequences through trained LSTM
7. **Constraints** - Check and enforce [Low, High] bounds
8. **Output** - Save predictions and visualizations

### Outputs
- `predictions_lstm.csv`
- `results_lstm.png`

---

## Common Elements

### Feature engineering

All models use these 8 technical indicators:

- **RSI (5-period)**: Measures momentum, shows if price is overbought (>70) or oversold (<30)
- **MACD (5/10)**: Difference between fast and slow moving averages, indicates trend direction
- **ROC (3-period)**: Rate of price change over 3 periods
- **ATR (5-period)**: Average true range normalized by current price, measures volatility
- **HL_Range**: Daily high-low spread relative to opening price
- **Volume_Change**: Percentage change in trading volume
- **Close_lag1_pct**: Previous day's return
- **Close_lag2_pct**: Two-day lagged return

These features capture different aspects: momentum, trend, volatility, and recent price action.

### Why predict returns instead of prices?

The training and test data might have different absolute price levels (train around $6000, test around $6200). Predicting percentage returns makes the model more robust to these shifts since returns are relative measures. After predicting the return, we convert back to a price: predicted_close = predicted_return × open + open.

### Preprocessing steps

1. Calculate all technical indicators from raw OHLCV data
2. Fill any NaN values (forward fill, then backward fill for any remaining)
3. Apply StandardScaler to features and target (zero mean, unit variance)
4. Train the model on scaled data
5. Inverse transform predictions back to the original scale
6. Convert predicted returns to closing prices

### Constraint enforcement

The competition requires predictions to fall within each day's [Low, High] range. We handle this by:

1. Making unconstrained predictions first
2. Counting violations (predictions outside valid range) for analysis
3. Clipping predictions to enforce constraints: `np.clip(pred, Low, High)`

Tracking violations before clipping tells us how naturally the model respects market constraints.

---

## Model Comparison

**VQC** uses a quantum circuit to directly learn feature representations. The entanglement between qubits can capture complex correlations. Training takes 10-15 minutes on CPU.

**CNN+QGAF** treats quantum-encoded time series as images and uses convolutions to detect local patterns. Good for recognizing short-term shapes in the data. Training takes 15-20 minutes.

**LSTM+QGAF** combines quantum encoding with sequential modeling. The LSTM's memory cells can learn long-term dependencies in the quantum-enhanced features. Training takes 20-25 minutes.

Each approach has different strengths. VQC is parameter-efficient with direct quantum learning. CNN excels at spatial pattern recognition. LSTM handles long sequences well. We tested all three to find which worked best for this dataset.

---

## Running the Code

1. Install requirements: `pip install pandas numpy pennylane tensorflow scikit-learn matplotlib jupyterlab`
2. Make sure `X_train.csv` and `X_test.csv` are in the same folder as the notebooks
3. Open Jupyter Lab and select a notebook
4. Run all cells in order

Each notebook is self-contained and will produce a predictions CSV file and a visualization showing how well the predictions fit within the constraints.

The training data needs columns: Date, Open, High, Low, Close, Volume
The test data needs: Date, Open, High, Low, Volume (we estimate Close for feature engineering)

---

## Technical Details

**Evaluation metrics:**
- MSE (Mean Squared Error): Average of squared differences, lower is better
- R²: Proportion of variance explained, 1.0 is perfect, higher is better
- Violation rate: Percentage of predictions outside [Low, High] before clipping

**Training configuration:**
- Optimizer: Adam with default learning rate
- Loss function: Mean Squared Error
- Batch size: 32
- Epochs: 50-100 with early stopping to prevent overfitting
- Random seeds set to 42 for reproducibility

**Hardware:**
- Works fine on CPU (10-25 min per model)
- GPU speeds things up to 3-8 minutes if available
- Needs 8GB+ RAM

---

## What We Learned

Predicting returns rather than absolute prices was crucial for generalizing across different price regimes. The quantum feature extraction (either through VQC or QGAF) gave us an edge over purely classical approaches. Testing multiple architectures let us pick the best performer rather than committing to a single approach.

The constraint violations metric proved useful for understanding model behavior - low violation rates mean the model naturally learned to respect market bounds rather than relying heavily on post-processing clipping.
