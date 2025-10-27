import pandas as pd
import numpy as np
import pennylane as qml
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

# ============================================================================
# FEATURE ENGINEERING FUNCTIONS
# ============================================================================

def calculate_rsi(close, period=5):
    """Relative Strength Index - momentum oscillator"""
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd_short(close):
    """MACD with shorter periods (5, 10) for crypto volatility"""
    ema5 = close.ewm(span=5, adjust=False).mean()
    ema10 = close.ewm(span=10, adjust=False).mean()
    return ema5 - ema10

def calculate_roc(close, period=3):
    """Rate of Change - percentage price change"""
    return (close - close.shift(period)) / close.shift(period) * 100

def calculate_atr(high, low, close, period=5):
    """Average True Range - volatility indicator"""
    high_low = high - low
    high_close = np.abs(high - close.shift())
    low_close = np.abs(low - close.shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return true_range.rolling(window=period).mean()

def add_features(df):
    """Engineer 8 technical indicators from OHLCV data"""
    df['RSI'] = calculate_rsi(df['Close'], period=5)
    df['MACD'] = calculate_macd_short(df['Close'])
    df['ROC'] = calculate_roc(df['Close'], period=3)
    df['ATR'] = calculate_atr(df['High'], df['Low'], df['Close'], period=5) / df['Close'] * 100
    df['HL_Range'] = (df['High'] - df['Low']) / df['Open']
    df['Volume_Change'] = df['Volume'].pct_change()
    df['Close_lag1_pct'] = (df['Close'].shift(1) - df['Open']) / df['Open']
    df['Close_lag2_pct'] = (df['Close'].shift(2) - df['Open'].shift(1)) / df['Open'].shift(1)
    return df

# ============================================================================
# QUANTUM CIRCUIT DEFINITION
# ============================================================================

n_qubits = 3
dev = qml.device('default.qubit', wires=n_qubits)

@qml.qnode(dev, interface='tf')
def vqc_circuit(inputs, weights):
    """
    Variational Quantum Circuit with:
    - Angle encoding (RY gates) for input features
    - 3 variational layers with RY/RZ rotations
    - CNOT entanglement between adjacent qubits
    """
    # Angle encoding
    for i in range(n_qubits):
        qml.RY(inputs[i % tf.shape(inputs)[0]], wires=i)
    
    # Variational layers
    for layer in range(3):
        for i in range(n_qubits):
            qml.RY(weights[layer, i, 0], wires=i)
            qml.RZ(weights[layer, i, 1], wires=i)
        for i in range(n_qubits - 1):
            qml.CNOT(wires=[i, i + 1])
    
    return qml.expval(qml.PauliZ(0))

class QuantumLayerVQC(keras.layers.Layer):
    """Custom Keras layer wrapping the quantum circuit"""
    def __init__(self, n_qubits):
        super().__init__()
        self.n_qubits = n_qubits
        # Trainable quantum weights: 3 layers × 3 qubits × 2 parameters (RY, RZ)
        self.q_weights = self.add_weight(
            shape=(3, n_qubits, 2), 
            initializer='random_normal', 
            trainable=True
        )
    
    def call(self, inputs):
        """Execute quantum circuit for each sample"""
        def single_sample(x):
            return tf.cast(vqc_circuit(x, self.q_weights), tf.float32)
        return tf.expand_dims(tf.map_fn(single_sample, inputs, dtype=tf.float32), axis=-1)

# ============================================================================
# DATA LOADING AND PREPROCESSING
# ============================================================================

print("Loading training data...")
df_train = pd.read_csv('X_train.csv')
df_train = add_features(df_train).ffill().bfill()  # Fill NaN from indicators

features = ['RSI', 'MACD', 'ROC', 'ATR', 'HL_Range', 'Volume_Change', 'Close_lag1_pct', 'Close_lag2_pct']
X_train = df_train[features].values

# Target: predict returns (Close - Open) / Open
y_train_returns = ((df_train['Close'] - df_train['Open']) / df_train['Open']).values

# Normalize features and target
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)

scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train_returns.reshape(-1, 1)).ravel()

# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

inputs = keras.Input(shape=(8,))
quantum_out = QuantumLayerVQC(n_qubits)(inputs)  # Quantum feature extraction
dense1 = keras.layers.Dense(16, activation='relu')(quantum_out)
dense2 = keras.layers.Dense(8, activation='relu')(dense1)
output = keras.layers.Dense(1)(dense2)  # Predict normalized return

model = keras.Model(inputs=inputs, outputs=output)
model.compile(optimizer='adam', loss='mse')

# ============================================================================
# TRAINING
# ============================================================================

print("Training VQC...")
model.fit(
    X_train_scaled, y_train_scaled, 
    epochs=100, 
    batch_size=32, 
    verbose=1,
    callbacks=[keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True)]
)

model.save('vqc_final.keras')

# ============================================================================
# PREDICTION ON TEST DATA
# ============================================================================

print("\nPredicting...")
df_test = pd.read_csv('X_test.csv')
df_test['Close'] = (df_test['High'] + df_test['Low']) / 2  # Estimate Close for feature engineering
df_test = add_features(df_test).ffill().bfill()

X_test = df_test[features].values
X_test_scaled = scaler_X.transform(X_test)

# Extract constraint bounds
open_test = df_test['Open'].values
low_test = df_test['Low'].values
high_test = df_test['High'].values

# Predict normalized returns and convert back to prices
y_pred_returns_scaled = model.predict(X_test_scaled, verbose=0)
y_pred_returns = scaler_y.inverse_transform(y_pred_returns_scaled).ravel()
y_pred = y_pred_returns * open_test + open_test

# ============================================================================
# VIOLATION ANALYSIS
# ============================================================================

violations = (y_pred < low_test) | (y_pred > high_test)
violations_count = np.sum(violations)

print(f"\n📊 Constraint Analysis:")
print(f"Violations before clipping: {violations_count}/{len(violations)}")
print(f"Violation rate: {violations_count/len(violations)*100:.1f}%")

# Clip predictions to satisfy constraints
y_pred_final = np.clip(y_pred, low_test, high_test)

# ============================================================================
# SAVE SUBMISSION
# ============================================================================

submission = pd.DataFrame({'Date': df_test['Date'], 'Close': y_pred_final})
submission.to_csv('predictions_vqc.csv', index=False)

print(f"\n✅ {len(y_pred_final)} predictions saved to 'predictions_vqc.csv'")
print(f"Range: [{y_pred_final.min():.2f}, {y_pred_final.max():.2f}]")
print(submission)

# ============================================================================
# VISUALIZATION
# ============================================================================

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

# Plot 1: Predictions with constraint bounds
ax1.plot(y_pred_final, label='Predicted Close', color='blue', linewidth=2)
ax1.plot(low_test, label='Low Bound', color='green', linestyle='--', alpha=0.7)
ax1.plot(high_test, label='High Bound', color='red', linestyle='--', alpha=0.7)
ax1.scatter(np.where(violations)[0], y_pred[violations], 
           color='red', s=50, zorder=5, label=f'Violations ({np.sum(violations)})')
ax1.fill_between(range(len(low_test)), low_test, high_test, alpha=0.2, color='gray')
ax1.set_title('VQC Predictions vs Constraint Bounds')
ax1.set_xlabel('Sample Index')
ax1.set_ylabel('Price')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Violation summary
ax2.bar(['Within Range', 'Violations'], 
        [np.sum(~violations), np.sum(violations)],
        color=['green', 'red'])
ax2.set_title(f'Constraint Violations: {np.sum(violations)}/{len(violations)} ({np.sum(violations)/len(violations)*100:.1f}%)')
ax2.set_ylabel('Count')
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('prediction_violations.png', dpi=300)
print("\n📊 Visualization saved as 'prediction_violations.png'")
plt.show()
