"""
BabyWatch Phase 2 - TinyML Cry Detection Training
Dataset: Infant Cry Dataset (Kaggle)
Output: baby_cry_model.tflite (for ESP32 deployment)
"""

import os
import librosa
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from pathlib import Path

# ===== CONFIGURATION =====
DATASET_PATH = "./data/infant_cry_dataset"  # Adjust path based on your structure
SR = 16000  # Sample rate
DURATION = 1.0  # 1 second audio windows
N_MFCC = 13  # MFCC coefficients
TEST_SIZE = 0.2
VALIDATION_SPLIT = 0.2
EPOCHS = 300
BATCH_SIZE = 8
LEARNING_RATE = 0.00005

print("🚀 BabyWatch TinyML Training Pipeline")
print("=" * 50)

# ===== STEP 1: EXPLORE DATASET =====
print("\n📂 Exploring dataset structure...")

cry_files = []
noise_files = []

# Common dataset structures:
# Option A: cry/ and noise/ folders
if os.path.exists(os.path.join(DATASET_PATH, "cry")):
    cry_files = [f for f in os.listdir(os.path.join(DATASET_PATH, "cry")) 
                 if f.endswith(('.wav', '.mp3', '.ogg'))]
    
if os.path.exists(os.path.join(DATASET_PATH, "noise")):
    noise_files = [f for f in os.listdir(os.path.join(DATASET_PATH, "noise")) 
                   if f.endswith(('.wav', '.mp3', '.ogg'))]

# Option B: CSV metadata file
if os.path.exists(os.path.join(DATASET_PATH, "metadata.csv")):
    import pandas as pd
    df = pd.read_csv(os.path.join(DATASET_PATH, "metadata.csv"))
    print(f"Found metadata.csv with {len(df)} samples")
    print(df.head())

print(f"Found {len(cry_files)} cry files")
print(f"Found {len(noise_files)} noise files")

# ===== STEP 2: FEATURE EXTRACTION FUNCTION =====
print("\n🎵 Extracting MFCC features...")

def extract_mfcc(audio_path, sr=SR, n_mfcc=N_MFCC, duration=DURATION):
    """
    Extract MFCC features from audio file
    Returns: averaged MFCC vector (n_mfcc,)
    """
    try:
        # Load audio
        audio, _ = librosa.load(audio_path, sr=sr, duration=duration)
        
        # Pad or trim to exact duration
        samples_required = int(sr * duration)
        if len(audio) < samples_required:
            audio = np.pad(audio, (0, samples_required - len(audio)))
        else:
            audio = audio[:samples_required]
        
        # Extract MFCC
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
        
        # Return mean across time axis (shape: n_mfcc,)
        return np.mean(mfcc.T, axis=0)
    
    except Exception as e:
        print(f"❌ Error processing {audio_path}: {e}")
        return None

# ===== STEP 3: LOAD DATA =====
print("\n📥 Loading and processing audio files...")

X = []
y = []

# Load crying samples (label = 1)
if cry_files:
    print(f"Loading {len(cry_files)} cry samples...")
    for i, file in enumerate(cry_files):
        file_path = os.path.join(DATASET_PATH, "cry", file)
        mfcc_features = extract_mfcc(file_path)
        
        if mfcc_features is not None:
            X.append(mfcc_features)
            y.append(1)  # Cry = 1
            
            if (i + 1) % 10 == 0:
                print(f"  ✓ Processed {i + 1}/{len(cry_files)} cry files")

# Load noise/non-cry samples (label = 0)
if noise_files:
    print(f"Loading {len(noise_files)} noise samples...")
    for i, file in enumerate(noise_files):
        file_path = os.path.join(DATASET_PATH, "noise", file)
        mfcc_features = extract_mfcc(file_path)
        
        if mfcc_features is not None:
            X.append(mfcc_features)
            y.append(0)  # Noise = 0
            
            if (i + 1) % 10 == 0:
                print(f"  ✓ Processed {i + 1}/{len(noise_files)} noise files")

X = np.array(X)
y = np.array(y)

print(f"\n✅ Loaded {len(X)} samples")
print(f"   Feature shape: {X.shape}")
print(f"   Cry samples (1): {np.sum(y)}")
print(f"   Noise samples (0): {len(y) - np.sum(y)}")

if len(X) < 10:
    print("⚠️  Warning: Very few samples. Consider downloading full dataset.")

# ===== STEP 4: TRAIN-TEST SPLIT =====
print("\n🔀 Splitting data (80% train, 20% test)...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=42, stratify=y
)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Training set: {len(X_train)} samples")
print(f"Test set: {len(X_test)} samples")

# ===== STEP 5: BUILD MODEL =====
print("\n🧠 Building neural network...")

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(N_MFCC,)),
    
    # First dense layer with dropout
    tf.keras.layers.Dense(64, activation='relu', name='dense_1'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.5),
    
    # Second dense layer
    tf.keras.layers.Dense(32, activation='relu', name='dense_2'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),
    
    # Output layer (binary classification)
    tf.keras.layers.Dense(1, activation='sigmoid', name='output')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
)

print(model.summary())

# ===== STEP 6: TRAIN MODEL =====
print("\n🚂 Training model (this may take 5-10 minutes)...")

history = model.fit(
    X_train_scaled, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=VALIDATION_SPLIT,
    verbose=1,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=30,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-7
        )
    ]
)

# ===== STEP 7: EVALUATE =====
print("\n📊 Evaluating on test set...")

test_loss, test_accuracy, test_precision, test_recall = model.evaluate(X_test_scaled, y_test)

print(f"\n✅ Test Results:")
print(f"   Accuracy:  {test_accuracy * 100:.2f}%")
print(f"   Precision: {test_precision * 100:.2f}%")
print(f"   Recall:    {test_recall * 100:.2f}%")
print(f"   Loss:      {test_loss:.4f}")

# ===== STEP 8: PLOT TRAINING HISTORY =====
print("\n📈 Saving training plots...")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Accuracy
axes[0].plot(history.history['accuracy'], label='Train Accuracy')
axes[0].plot(history.history['val_accuracy'], label='Val Accuracy')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].set_title('Model Accuracy')
axes[0].legend()
axes[0].grid(True)

# Loss
axes[1].plot(history.history['loss'], label='Train Loss')
axes[1].plot(history.history['val_loss'], label='Val Loss')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].set_title('Model Loss')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig('training_history.png', dpi=150)
print("   ✓ Saved: training_history.png")

# ===== STEP 9: CONVERT TO TFLITE =====
print("\n🔄 Converting to TensorFlow Lite...")

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
]
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Quantization (reduces model size for ESP32)
def representative_dataset():
    for i in range(min(100, len(X_train_scaled))):
        yield [X_train_scaled[i:i+1].astype(np.float32)]

converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8

try:
    tflite_model = converter.convert()
except Exception as e:
    print(f"⚠️  Quantization failed, using default conversion: {e}")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

# Save model
model_path = "baby_cry_model.tflite"
with open(model_path, 'wb') as f:
    f.write(tflite_model)

model_size = os.path.getsize(model_path) / 1024
print(f"✅ Model saved: {model_path}")
print(f"   Size: {model_size:.2f} KB")

# ===== STEP 10: TEST TFLITE MODEL =====
print("\n🧪 Testing TFLite model on sample data...")

interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(f"   Input shape: {input_details[0]['shape']}")
print(f"   Output shape: {output_details[0]['shape']}")

# Test on first 5 samples
print("\n   Sample predictions:")
for i in range(min(5, len(X_test_scaled))):
    # Prepare input
    test_sample = np.expand_dims(X_test_scaled[i], axis=0)
    
    # Quantize if needed
    if input_details[0]['dtype'] == np.uint8:
        input_scale, input_zero_point = input_details[0]['quantization']
        test_sample = (test_sample / input_scale + input_zero_point).astype(np.uint8)
    else:
        test_sample = test_sample.astype(np.float32)
    
    # Run inference
    interpreter.set_tensor(input_details[0]['index'], test_sample)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    
    # Dequantize if needed
    if output_details[0]['dtype'] == np.uint8:
        output_scale, output_zero_point = output_details[0]['quantization']
        cry_prob = (output - output_zero_point) * output_scale
    else:
        cry_prob = output[0]
    
    true_label = "Cry" if y_test[i] == 1 else "Noise"
    pred_label = "Cry" if cry_prob > 0.5 else "Noise"
    
    print(f"   Sample {i+1}: True={true_label:5} | Pred={pred_label:5} | Prob={cry_prob[0]:.3f}")

# ===== STEP 11: GENERATE C++ HEADER (for Arduino) =====
print("\n📝 Generating C++ header for Arduino...")

# Read model as hex
with open(model_path, 'rb') as f:
    model_data = f.read()

# Create header file
header_content = f"""// baby_cry_model_data.h
// Auto-generated TFLite model header for BabyWatch Phase 2
// Model size: {len(model_data)} bytes
// Generated: {pd.Timestamp.now()}

const unsigned char baby_cry_model_data[] = {{
"""

# Add hex data in chunks
for i in range(0, len(model_data), 12):
    chunk = model_data[i:i+12]
    hex_str = ', '.join(f'0x{b:02x}' for b in chunk)
    header_content += f"  {hex_str},\n"

header_content += f"""
}};
const int baby_cry_model_data_len = {len(model_data)};
"""

header_path = "baby_cry_model_data.h"
with open(header_path, 'w') as f:
    f.write(header_content)

print(f"✅ Header generated: {header_path}")

# ===== SUMMARY =====
print("\n" + "=" * 50)
print("🎉 TRAINING COMPLETE!")
print("=" * 50)
print(f"\n📦 Outputs generated:")
print(f"   1. baby_cry_model.tflite     (Deploy to ESP32)")
print(f"   2. baby_cry_model_data.h     (Use in Arduino)")
print(f"   3. training_history.png      (Training graphs)")
print(f"\n📊 Model Performance:")
print(f"   Test Accuracy:  {test_accuracy * 100:.2f}%")
print(f"   Model Size:     {model_size:.2f} KB (fits on ESP32)")
print(f"\n🚀 Next Steps:")
print(f"   1. Copy baby_cry_model_data.h to Arduino sketch")
print(f"   2. Use TFLite Micro to run inference on ESP32")
print(f"   3. Integrate with your audio sampling pipeline")
print("\n" + "=" * 50)
