import pandas as pd
import numpy as np
import pennylane as qml
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
import tensorflow as tf
from tensorflow import keras
from tqdm import tqdm
import time
import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

np.random.seed(42)
tf.random.set_seed(42)

# ===== Feature Engineering =====

def calculate_rsi(close, period=5):
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd_short(close):
    ema5 = close.ewm(span=5, adjust=False).mean()
    ema10 = close.ewm(span=10, adjust=False).mean()
    return ema5 - ema10

def calculate_roc(close, period=3):
    return (close - close.shift(period)) / close.shift(period) * 100

def calculate_atr(high, low, close, period=5):
    high_low = high - low
    high_close = np.abs(high - close.shift())
    low_close = np.abs(low - close.shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return true_range.rolling(window=period).mean()

def add_features(df):
    # Use percentage-based features
    df['RSI'] = calculate_rsi(df['Close'], period=5)
    df['MACD'] = calculate_macd_short(df['Close'])
    df['ROC'] = calculate_roc(df['Close'], period=3)
    df['ATR'] = calculate_atr(df['High'], df['Low'], df['Close'], period=5) / df['Close'] * 100  # Normalized
    df['HL_Range'] = (df['High'] - df['Low']) / df['Open']
    df['Volume_Change'] = df['Volume'].pct_change()
    df['Close_lag1_pct'] = (df['Close'].shift(1) - df['Open']) / df['Open']
    df['Close_lag2_pct'] = (df['Close'].shift(2) - df['Open'].shift(1)) / df['Open'].shift(1)
    return df

def create_sequences(X, y, seq_len):
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_len):
        X_seq.append(X[i:i+seq_len])
        y_seq.append(y[i+seq_len])
    return np.array(X_seq), np.array(y_seq)

def apply_constraints(y_pred, low, high):
    return np.clip(y_pred, low, high)

# ===== Load Data =====

print("Loading data...")
df = pd.read_csv('X_train.csv')
split_idx = int(len(df) * 0.85)
df_train = add_features(df[:split_idx].copy()).ffill().bfill()
df_val = add_features(df[split_idx:].copy()).ffill().bfill()

features = ['RSI', 'MACD', 'ROC', 'ATR', 'HL_Range', 'Volume_Change', 'Close_lag1_pct', 'Close_lag2_pct']
X_train = df_train[features].values
X_val = df_val[features].values

# Predict returns: (Close - Open) / Open
y_train_returns = ((df_train['Close'] - df_train['Open']) / df_train['Open']).values
y_val_returns = ((df_val['Close'] - df_val['Open']) / df_val['Open']).values


open_train = df_train['Open'].values
open_val = df_val['Open'].values
low_val = df_val['Low'].values
high_val = df_val['High'].values
y_val_actual = df_val['Close'].values

scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_val_scaled = scaler_X.transform(X_val)

scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train_returns.reshape(-1, 1)).ravel()
y_val_scaled = scaler_y.transform(y_val_returns.reshape(-1, 1)).ravel()

results = []


# ===== MODEL 1: Linear Regression =====

print("\n" + "="*70)
print("MODEL 1: Linear Regression (Baseline)")
print("="*70)

lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train_returns)
y_pred_returns_lr = lr_model.predict(X_val_scaled)
y_pred_lr = y_pred_returns_lr * open_val + open_val  # Convert back to Close
y_pred_lr_clipped = apply_constraints(y_pred_lr, low_val, high_val)

violations_lr = np.sum((y_pred_lr < low_val) | (y_pred_lr > high_val))
print(f"Violations: {violations_lr}/{len(y_pred_lr)}")

mse_lr = mean_squared_error(y_val_actual, y_pred_lr_clipped)
rmse_lr = np.sqrt(mse_lr)
r2_lr = r2_score(y_val_actual, y_pred_lr_clipped)

print(f"RMSE: {rmse_lr:.2f}, R²: {r2_lr:.4f}")
results.append({'Model': 'Linear Regression', 'RMSE': rmse_lr, 'R2': r2_lr, 'Violations': violations_lr})

# ===== MODEL 2: Quantum Kernel SVM ====
print("\n" + "="*70)
print("MODEL 2: Quantum Kernel SVM")
print("="*70)

n_qubits = 3
dev_qsvm = qml.device('default.qubit', wires=n_qubits)

@qml.qnode(dev_qsvm)
def quantum_kernel_circuit(x1, x2):
    qml.AmplitudeEmbedding(x1, wires=range(n_qubits), normalize=True, pad_with=0.0)
    qml.adjoint(qml.AmplitudeEmbedding)(x2, wires=range(n_qubits), normalize=True, pad_with=0.0)
    return qml.probs(wires=range(n_qubits))

def quantum_kernel(X1, X2):
    n1, n2 = len(X1), len(X2)
    K = np.zeros((n1, n2))
    for i in tqdm(range(n1), desc="Computing kernel"):
        for j in range(n2):
            K[i, j] = np.sum(quantum_kernel_circuit(X1[i], X2[j]))
    return K

start = time.time()
K_train = quantum_kernel(X_train_scaled, X_train_scaled)
svr = SVR(kernel='precomputed', C=1.0)
svr.fit(K_train, y_train_returns)
K_val = quantum_kernel(X_val_scaled, X_train_scaled)
y_pred_returns_qsvm = svr.predict(K_val)
y_pred_qsvm = y_pred_returns_qsvm * open_val + open_val
y_pred_qsvm_clipped = apply_constraints(y_pred_qsvm, low_val, high_val)

violations_qsvm = np.sum((y_pred_qsvm < low_val) | (y_pred_qsvm > high_val))
print(f"Violations: {violations_qsvm}/{len(y_pred_qsvm)}")

mse_qsvm = mean_squared_error(y_val_actual, y_pred_qsvm_clipped)
rmse_qsvm = np.sqrt(mse_qsvm)
r2_qsvm = r2_score(y_val_actual, y_pred_qsvm_clipped)
time_qsvm = time.time() - start

print(f"RMSE: {rmse_qsvm:.2f}, R²: {r2_qsvm:.4f}, Time: {time_qsvm:.1f}s")
results.append({'Model': 'Quantum Kernel SVM', 'RMSE': rmse_qsvm, 'R2': r2_qsvm, 'Violations': violations_qsvm})

# ===== MODEL 3: Quantum LSTM with Sliding Window ===== 
print("\n" + "="*70)
print("MODEL 3: Quantum LSTM (Sliding Window)")
print("="*70)
sliding_windows = [3, 5, 7, 10]
best_rmse = np.inf
best_metrics = {}

for seq_len in sliding_windows:
    print(f"\nEvaluating seq_len = {seq_len}")
    X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_scaled, seq_len)
    X_val_seq, y_val_seq = create_sequences(X_val_scaled, y_val_scaled, seq_len)
    open_val_seq = open_val[seq_len:]
    low_val_seq = low_val[seq_len:]
    high_val_seq = high_val[seq_len:]
    y_val_seq_actual = y_val_actual[seq_len:]

    dev_qlstm = qml.device('default.qubit', wires=n_qubits)

    @qml.qnode(dev_qlstm, interface='tf')
    def quantum_circuit_deep(inputs, weights):
        qml.AmplitudeEmbedding(features=inputs, wires=range(n_qubits), normalize=True, pad_with=0.0)
        for layer in range(4):
            for i in range(n_qubits):
                qml.RY(weights[layer, i, 0], wires=i)
            for i in range(n_qubits - 1):
                qml.CNOT(wires=[i, i+1])
            for i in range(n_qubits):
                qml.RZ(weights[layer, i, 1], wires=i)
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    class QuantumLayerDeep(keras.layers.Layer):
        def __init__(self, n_qubits):
            super().__init__()
            self.n_qubits = n_qubits
            self.q_weights = self.add_weight(shape=(4, n_qubits, 2), initializer='random_normal', trainable=True)

        def call(self, inputs):
            def process_sequence(seq):
                def process_timestep(features):
                    q_out = quantum_circuit_deep(features, self.q_weights)
                    return tf.cast(tf.stack(q_out), tf.float32)
                return tf.map_fn(process_timestep, seq, dtype=tf.float32)
            return tf.map_fn(process_sequence, inputs, dtype=tf.float32)

        def compute_output_shape(self, input_shape):
            return (input_shape[0], input_shape[1], self.n_qubits)

    inputs_qlstm = keras.Input(shape=(seq_len, 8))
    quantum_out = QuantumLayerDeep(n_qubits)(inputs_qlstm)
    lstm_out = keras.layers.LSTM(32)(quantum_out)
    outputs_qlstm = keras.layers.Dense(1)(lstm_out)
    qlstm_model = keras.Model(inputs=inputs_qlstm, outputs=outputs_qlstm)
    qlstm_model.compile(optimizer='adam', loss='mse')

    start = time.time()
    qlstm_model.fit(
        X_train_seq, y_train_seq, epochs=50, batch_size=32, verbose=0,
        validation_data=(X_val_seq, y_val_seq),
        callbacks=[keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)]
    )

    y_pred_returns_qlstm_scaled = qlstm_model.predict(X_val_seq, verbose=0)
    y_pred_returns_qlstm = scaler_y.inverse_transform(y_pred_returns_qlstm_scaled).ravel()
    y_pred_qlstm = y_pred_returns_qlstm * open_val_seq + open_val_seq
    y_pred_qlstm_clipped = apply_constraints(y_pred_qlstm, low_val_seq, high_val_seq)

    violations_qlstm = np.sum((y_pred_qlstm < low_val_seq) | (y_pred_qlstm > high_val_seq))
    mse_qlstm = mean_squared_error(y_val_seq_actual, y_pred_qlstm_clipped)
    rmse_qlstm = np.sqrt(mse_qlstm)
    r2_qlstm = r2_score(y_val_seq_actual, y_pred_qlstm_clipped)
    time_qlstm = time.time() - start

    print(f"Seq_len={seq_len}: RMSE={rmse_qlstm:.2f}, R²={r2_qlstm:.4f}, Violations={violations_qlstm}")

    # Save best sequence metrics
    if rmse_qlstm < best_rmse:
        best_rmse = rmse_qlstm
        best_metrics = {
            'RMSE': rmse_qlstm,
            'R2': r2_qlstm,
            'Violations': violations_qlstm,
            'seq_len': seq_len
        }

print(f"\n✅ Best QLSTM seq_len = {best_metrics['seq_len']} with RMSE={best_metrics['RMSE']:.2f}")

results.append({'Model': 'Quantum LSTM',
                'RMSE': best_metrics['RMSE'],
                'R2': best_metrics['R2'],
                'Violations': best_metrics['Violations']})


# ===== MODEL 4: QGAN =====
print("\n" + "="*70)
print("MODEL 4: QGAN")
print("="*70)

scaler_X_qgan = MinMaxScaler(feature_range=(0, 2*np.pi))
scaler_y_qgan = MinMaxScaler(feature_range=(0, 2*np.pi))
X_train_qgan = scaler_X_qgan.fit_transform(X_train)
X_val_qgan = scaler_X_qgan.transform(X_val)
y_train_qgan = scaler_y_qgan.fit_transform(y_train_returns.reshape(-1, 1)).ravel()

n_qubits_gan = 4
dev_gen = qml.device('default.qubit', wires=n_qubits_gan)
dev_disc = qml.device('default.qubit', wires=n_qubits_gan)

def gen_circuit(noise_feat, weights):
    for i in range(n_qubits_gan):
        qml.RY(noise_feat[i], wires=i)
    for layer in range(3):
        for i in range(n_qubits_gan):
            qml.RY(weights[layer, i, 0], wires=i)
            qml.RZ(weights[layer, i, 1], wires=i)
        for i in range(n_qubits_gan-1):
            qml.CNOT(wires=[i, i+1])
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits_gan)]

def disc_circuit(feat_price, weights):
    for i in range(n_qubits_gan):
        qml.RY(feat_price[i], wires=i)
    for layer in range(3):
        for i in range(n_qubits_gan):
            qml.RY(weights[layer, i, 0], wires=i)
            qml.RZ(weights[layer, i, 1], wires=i)
        for i in range(n_qubits_gan-1):
            qml.CNOT(wires=[i, i+1])
    return qml.expval(qml.PauliZ(0))

@qml.qnode(dev_gen, interface='autograd')
def gen_qnode(inputs, weights):
    return gen_circuit(inputs, weights)

@qml.qnode(dev_disc, interface='autograd')
def disc_qnode(inputs, weights):
    return disc_circuit(inputs, weights)

class Generator:
    def __init__(self):
        self.weights = np.random.randn(3, n_qubits_gan, 2) * 0.1
    
    def forward(self, noise, features):
        outputs = []
        for i in range(len(features)):
            combined = np.concatenate([noise[i], features[i][:n_qubits_gan-len(noise[i])]])
            outputs.append(np.mean(gen_qnode(combined, self.weights)))
        return np.array(outputs).reshape(-1, 1)

class Discriminator:
    def __init__(self):
        self.weights = np.random.randn(3, n_qubits_gan, 2) * 0.1
    
    def forward(self, features, prices):
        outputs = []
        for i in range(len(features)):
            combined = np.concatenate([features[i][:n_qubits_gan-1], [prices[i]]])
            outputs.append(1 / (1 + np.exp(-disc_qnode(combined, self.weights))))
        return np.array(outputs).reshape(-1, 1)

generator = Generator()
discriminator = Discriminator()
lr_gan = 0.01

start = time.time()
for epoch in tqdm(range(50), desc="Training QGAN"):
    noise = np.random.uniform(0, 2*np.pi, (len(X_train_qgan), 2))
    fake_y = generator.forward(noise, X_train_qgan)
    real_pred = discriminator.forward(X_train_qgan, y_train_qgan)
    fake_pred = discriminator.forward(X_train_qgan, fake_y.ravel())
    d_loss = -np.mean(np.log(real_pred + 1e-8) + np.log(1 - fake_pred + 1e-8))
    
    noise = np.random.uniform(0, 2*np.pi, (len(X_train_qgan), 2))
    fake_y = generator.forward(noise, X_train_qgan)
    fake_pred = discriminator.forward(X_train_qgan, fake_y.ravel())
    g_loss = -np.mean(np.log(fake_pred + 1e-8))
    
    generator.weights -= lr_gan * 0.01 * g_loss
    discriminator.weights -= lr_gan * 0.01 * d_loss

noise = np.random.uniform(0, 2*np.pi, (len(X_val_qgan), 2))
y_pred_qgan_scaled = generator.forward(noise, X_val_qgan)
y_pred_returns_qgan = scaler_y_qgan.inverse_transform(y_pred_qgan_scaled)
y_pred_qgan = y_pred_returns_qgan.ravel() * open_val + open_val

violations_qgan = np.sum((y_pred_qgan < low_val) | (y_pred_qgan > high_val))
print(f"Violations before clipping: {violations_qgan}/{len(y_pred_qgan)}")
y_pred_qgan_clipped = apply_constraints(y_pred_qgan, low_val, high_val)



mse_qgan = mean_squared_error(y_val_actual, y_pred_qgan_clipped)
rmse_qgan = np.sqrt(mse_qgan)
r2_qgan = r2_score(y_val_actual, y_pred_qgan_clipped)
time_qgan = time.time() - start

print(f"RMSE: {rmse_qgan:.2f}, R²: {r2_qgan:.4f}, Time: {time_qgan:.1f}s")
results.append({'Model': 'QGAN', 'RMSE': rmse_qgan, 'R2': r2_qgan, 'Violations': violations_qgan})


# ===== MODEL 5: Variational Quantum Circuit (VQC) =====

print("\n" + "="*70)
print("MODEL 5: Variational Quantum Circuit (VQC)")
print("="*70)

# Prepare data (scaled)
X_train_vqc, y_train_vqc = X_train_scaled, y_train_scaled
X_val_vqc, y_val_vqc = X_val_scaled, y_val_scaled

dev_vqc = qml.device('default.qubit', wires=n_qubits)

@qml.qnode(dev_vqc, interface='tf')
def vqc_circuit(inputs, weights):
    """Variational Quantum Circuit with symbolic tensor support."""
    n_features = tf.shape(inputs)[0]  
    # Encode features into qubits
    for i in range(n_qubits):
        qml.RY(inputs[i % n_features], wires=i)

    # Variational layers
    for layer in range(3):
        for i in range(n_qubits):
            qml.RY(weights[layer, i, 0], wires=i)
            qml.RZ(weights[layer, i, 1], wires=i)
        for i in range(n_qubits - 1):
            qml.CNOT(wires=[i, i + 1])

    # Output: expectation of PauliZ on first qubit
    return qml.expval(qml.PauliZ(0))


class QuantumLayerVQC(keras.layers.Layer):
    """Custom Keras layer wrapping the VQC."""
    def __init__(self, n_qubits):
        super().__init__()
        self.n_qubits = n_qubits
        self.q_weights = self.add_weight(
            shape=(3, n_qubits, 2),
            initializer='random_normal',
            trainable=True
        )

    def call(self, inputs):
        # inputs shape: (batch_size, n_features)
        def single_sample(x):
            out = vqc_circuit(x, self.q_weights)
            return tf.cast(out, tf.float32)

        # Apply VQC to each sample in the batch
        out = tf.map_fn(single_sample, inputs, dtype=tf.float32)
        # Ensure output is 2D for Dense layer: (batch_size, 1)
        out = tf.expand_dims(out, axis=-1)
        return out


# === Build hybrid model ===
inputs = keras.Input(shape=(X_train_vqc.shape[1],))
quantum_out = QuantumLayerVQC(n_qubits)(inputs)
dense1 = keras.layers.Dense(16, activation='relu')(quantum_out)
dense2 = keras.layers.Dense(8, activation='relu')(dense1)
output = keras.layers.Dense(1)(dense2)

vqc_model = keras.Model(inputs=inputs, outputs=output)
vqc_model.compile(optimizer='adam', loss='mse')

# === Training ===
start = time.time()
vqc_model.fit(
    X_train_vqc, y_train_vqc,
    epochs=50, batch_size=32, verbose=1,
    validation_data=(X_val_vqc, y_val_vqc),
    callbacks=[keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)]
)
vqc_model.save('vqc_model.keras')

# === Predict and evaluate ===
y_pred_returns_vqc_scaled = vqc_model.predict(X_val_vqc, verbose=0)
y_pred_returns_vqc = scaler_y.inverse_transform(y_pred_returns_vqc_scaled).ravel()
y_pred_vqc = y_pred_returns_vqc * open_val + open_val
y_pred_vqc_clipped = apply_constraints(y_pred_vqc, low_val, high_val)

violations_vqc = np.sum((y_pred_vqc < low_val) | (y_pred_vqc > high_val))
print(f"Violations: {violations_vqc}/{len(y_pred_vqc)}")

mse_vqc = mean_squared_error(y_val_actual, y_pred_vqc_clipped)
rmse_vqc = np.sqrt(mse_vqc)
r2_vqc = r2_score(y_val_actual, y_pred_vqc_clipped)
time_vqc = time.time() - start

print(f"RMSE: {rmse_vqc:.2f}, R²: {r2_vqc:.4f}, Time: {time_vqc:.1f}s")
results.append({
    'Model': 'VQC',
    'RMSE': rmse_vqc,
    'R2': r2_vqc,
    'Violations': violations_vqc
})

# ===== FINAL RESULTS =====

print("\n" + "="*70)
print("FINAL RESULTS")
print("="*70)
results_df = pd.DataFrame(results).sort_values('RMSE')
print(results_df.to_string(index=False))
print(f"\n🏆 Best: {results_df.iloc[0]['Model']} (RMSE: {results_df.iloc[0]['RMSE']:.2f}, R²: {results_df.iloc[0]['R2']:.4f}, Violations: {results_df.iloc[0]['Violations']:.6f})")
results_df.to_csv('final_results_returns.csv', index=False)

# ===== Update results_df for plotting =====
for i, row in enumerate(results_df['Model']):
    if "Quantum LSTM" in row:
        results_df.loc[i, 'RMSE'] = best_metrics['RMSE']
        results_df.loc[i, 'R2'] = best_metrics['R2']
        results_df.loc[i, 'Violations'] = best_metrics['Violations']
        results_df.loc[i, 'Best_Seq_Len'] = best_metrics['seq_len']  
        break

# Recompute MSE
results_df["MSE"] = results_df["RMSE"] ** 2

# ===== PLOTS =====

import matplotlib.pyplot as plt
metrics = ["RMSE", "R2", "Violations"]

# Individual metric comparison
fig, axes = plt.subplots(1, len(metrics), figsize=(16, 5))
for i, metric in enumerate(metrics):
    axes[i].bar(results_df['Model'], results_df[metric], color='skyblue', edgecolor='black')
    axes[i].set_title(metric, fontsize=14)
    axes[i].set_xticklabels(results_df['Model'], rotation=45, ha='right')
    if metric in ["RMSE"]:
        axes[i].set_yscale('log')
    axes[i].grid(axis='y', linestyle='--', alpha=0.7)
    # Annotate QLSTM best sequence
    for j, model in enumerate(results_df['Model']):
        if "Quantum LSTM" in model:
            axes[i].text(j, results_df.loc[j, metric]*1.05,
                         f"seq={int(results_df.loc[j,'Best_Seq_Len'])}",
                         ha='center', color='red', fontsize=10)

plt.suptitle("Model Performance Comparison (Log Scale for RMSE)", fontsize=16, fontweight='bold')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# Combined metrics
fig, ax = plt.subplots(figsize=(10, 6))
bar_width = 0.2
x = np.arange(len(results_df))

ax.bar(x - 0.3, results_df["RMSE"], width=bar_width, label="RMSE")
ax.bar(x - 0.1, results_df["MSE"], width=bar_width, label="MSE")
ax.bar(x + 0.1, results_df["R2"], width=bar_width, label="R²")
ax.bar(x + 0.3, results_df["Violations"], width=bar_width, label="Violations")

ax.set_yscale('log')
ax.set_xticks(x)
ax.set_xticklabels(results_df["Model"], rotation=45, ha='right')
ax.set_title("Overall Model Metric Comparison (Log Scale)", fontsize=16, fontweight='bold')
ax.legend()
ax.grid(axis='y', linestyle='--', alpha=0.7)

# Annotate best QLSTM sequence
for j, model in enumerate(results_df['Model']):
    if "Quantum LSTM" in model:
        ax.text(j + 0.3, results_df.loc[j,"Violations"]*1.05,
                f"seq={int(results_df.loc[j,'Best_Seq_Len'])}",
                ha='center', color='red', fontsize=10)

plt.tight_layout()
plt.show()