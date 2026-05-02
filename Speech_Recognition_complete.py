# -*- coding: utf-8 -*-
"""
Speech_Recognition_Complete.py
Single file — 2x augmentation + training + saves .h5
Run with: python Speech_Recognition_Complete.py
"""

"""1. Installing Dependencies"""
import subprocess
import sys
subprocess.check_call([sys.executable, "-m", "pip", "install",
    "kagglehub", "librosa", "tensorflow", "scikit-learn",
    "matplotlib", "seaborn", "pandas", "numpy"])

"""2. Importing Libraries"""
import kagglehub
import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Conv1D, MaxPooling1D, LSTM, Dense,
                                     Dropout, BatchNormalization,
                                     Input, Layer)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

"""3. Downloading Dataset"""
path = kagglehub.dataset_download("jesusrequena/mlend-spoken-numerals")
print("Dataset downloaded to:", path)

csv_path = os.path.join(path, 'MLEndSND_Audio_Attributes.csv')
df = pd.read_csv(csv_path)
audio_folder = os.path.join(path, 'MLEndSND_Public', 'MLEndSND_Public')

print("Unique emotions:", df['Intonation'].unique())
print("Emotion counts:\n", df['Intonation'].value_counts())

"""4. Defining AttentionLayer, MFCC and Augmentation Functions"""

class AttentionLayer(Layer):
    def build(self, input_shape):
        self.W = self.add_weight(shape=(input_shape[-1], 1),
                                  initializer='random_normal', trainable=True)
        super(AttentionLayer, self).build(input_shape)
    def call(self, x):
        score = tf.nn.tanh(tf.matmul(x, self.W))
        weights = tf.nn.softmax(score, axis=1)
        return tf.reduce_sum(x * weights, axis=1)

MAX_PAD_LEN = 174
N_MFCC = 40

def extract_mfcc_from_array(audio, sr, max_pad_len=MAX_PAD_LEN, n_mfcc=N_MFCC):
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    if mfcc.shape[1] < max_pad_len:
        mfcc = np.pad(mfcc, ((0, 0), (0, max_pad_len - mfcc.shape[1])))
    else:
        mfcc = mfcc[:, :max_pad_len]
    return mfcc

def extract_mfcc(file_path):
    try:
        audio, sr = librosa.load(file_path, sr=22050, duration=3.0)
        return extract_mfcc_from_array(audio, sr)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def augment_and_extract(audio, sr, label):
    results = []
    results.append((extract_mfcc_from_array(audio, sr), label))
    noise = np.random.randn(len(audio)) * 0.005
    results.append((extract_mfcc_from_array(audio + noise, sr), label))
    return results

print("✅ All functions defined.")

"""5. Loading Audio Files with 2x Augmentation (or from disk if already saved)"""

EMOTIONS = df['Intonation'].unique().tolist()

if os.path.exists('X_data.npy'):
    print("✅ Found saved data — loading from disk, skipping audio processing!")
    X = np.load('X_data.npy')
    y_cat = np.load('y_cat.npy')
    engagement_cat = np.load('engagement_cat.npy')
    with open('labels.pkl', 'rb') as f:
        labels = pickle.load(f)
    le = LabelEncoder()
    le.fit(labels)
    print("X shape:", X.shape)
    print("Classes:", le.classes_)

else:
    print("No saved data found — running full 2x augmentation...")
    features = []
    labels = []
    missing = 0

    for i, (_, row) in enumerate(df.iterrows()):
        fname = f"{int(row['Public filename']):05d}.wav"
        fpath = os.path.join(audio_folder, fname)
        if not os.path.exists(fpath):
            missing += 1
            continue
        try:
            audio, sr = librosa.load(fpath, sr=22050, duration=3.0)
            augmented = augment_and_extract(audio, sr, row['Intonation'])
            for mfcc, label in augmented:
                features.append(mfcc)
                labels.append(label)
        except Exception as e:
            print(f"Error: {e}")
            continue
        if (i+1) % 5000 == 0:
            print(f"Progress: {i+1}/32654 files...")

    print(f"\nTotal samples loaded: {len(features)}")
    print(f"Missing files: {missing}")

    X = np.array(features)
    X = X.reshape(X.shape[0], N_MFCC, MAX_PAD_LEN)
    X = np.transpose(X, (0, 2, 1))
    X = (X - X.mean()) / (X.std() + 1e-8)

    le = LabelEncoder()
    y_encoded = le.fit_transform(labels)
    y_cat = to_categorical(y_encoded)

    engagement_cat = to_categorical(
        [1 if e in ['excited', 'question'] else 0 for e in labels],
        num_classes=2)

    print("X shape:", X.shape)
    print("Classes:", le.classes_)

    np.save('X_data.npy', X)
    np.save('y_cat.npy', y_cat)
    np.save('engagement_cat.npy', engagement_cat)
    with open('labels.pkl', 'wb') as f:
        pickle.dump(labels, f)
    print("✅ Saved to disk!")

"""6. Train Test Split"""

X_train, X_test, y_train, y_test = train_test_split(
    X, y_cat, test_size=0.2, random_state=42, stratify=y_cat
)
_, _, eng_train, eng_test = train_test_split(
    X, engagement_cat, test_size=0.2, random_state=42, stratify=y_cat
)

print(f"Train samples: {X_train.shape[0]}")
print(f"Test samples:  {X_test.shape[0]}")

"""7. Building CNN + LSTM + Attention Dual Output Model"""

num_classes = len(EMOTIONS)

inputs = Input(shape=(MAX_PAD_LEN, N_MFCC))
x = Conv1D(64, kernel_size=5, activation='relu')(inputs)
x = BatchNormalization()(x)
x = MaxPooling1D(pool_size=2)(x)
x = Dropout(0.3)(x)
x = Conv1D(128, kernel_size=5, activation='relu')(x)
x = BatchNormalization()(x)
x = MaxPooling1D(pool_size=2)(x)
x = Dropout(0.3)(x)
x = LSTM(128, return_sequences=True)(x)
x = AttentionLayer()(x)
x = Dropout(0.4)(x)
shared = Dense(64, activation='relu')(x)
emotion_out = Dense(num_classes, activation='softmax', name='emotion')(shared)
engagement_out = Dense(2, activation='softmax', name='engagement')(shared)

dual_model = Model(inputs=inputs, outputs=[emotion_out, engagement_out])
dual_model.compile(
    optimizer='adam',
    loss={'emotion': 'categorical_crossentropy',
          'engagement': 'categorical_crossentropy'},
    metrics={'emotion': 'accuracy', 'engagement': 'accuracy'}
)
dual_model.summary()

"""8. Training the Model — 10 Epochs"""

callbacks = [
    EarlyStopping(monitor='val_emotion_accuracy', patience=10,
                  restore_best_weights=True, mode='max'),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
]

history = dual_model.fit(
    X_train,
    {'emotion': y_train, 'engagement': eng_train},
    validation_split=0.2,
    epochs=10,
    batch_size=32,
    callbacks=callbacks
)

"""9. Saving Training Curves"""

plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['emotion_accuracy'], label='Train Emotion Accuracy')
plt.plot(history.history['val_emotion_accuracy'], label='Val Emotion Accuracy')
plt.title('Emotion Accuracy Over Epochs')
plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['engagement_accuracy'], label='Train Engagement Accuracy')
plt.plot(history.history['val_engagement_accuracy'], label='Val Engagement Accuracy')
plt.title('Engagement Accuracy Over Epochs')
plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend()

plt.tight_layout()
plt.savefig('training_curves.png')
plt.close()
print("✅ Training curves saved as training_curves.png")

"""10. Evaluation — Classification Report"""

emotion_preds, engagement_preds = dual_model.predict(X_test)

y_pred = np.argmax(emotion_preds, axis=1)
y_true = np.argmax(y_test, axis=1)

print("=" * 50)
print("EMOTION CLASSIFICATION RESULTS")
print("=" * 50)
print(classification_report(y_true, y_pred, target_names=le.classes_))

eng_pred = np.argmax(engagement_preds, axis=1)
eng_true = np.argmax(eng_test, axis=1)
eng_acc = np.mean(eng_pred == eng_true) * 100
print(f"Engagement Detection Accuracy: {eng_acc:.2f}%")

"""11. Confusion Matrix"""

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.title('Emotion Confusion Matrix')
plt.ylabel('True Label'); plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.close()
print("✅ Confusion matrix saved as confusion_matrix.png")

"""12. Saving the Model and Labels"""

dual_model.save('ser_dual_model.h5')
print("✅ Model saved as ser_dual_model.h5")

with open('labels.pkl', 'wb') as f:
    pickle.dump(labels, f)
print("✅ Labels saved as labels.pkl")

print("\n" + "=" * 50)
print("ALL DONE!")
print("=" * 50)
print("Files saved in current directory:")
print("  ser_dual_model.h5   ← copy this to EOT/model/")
print("  labels.pkl          ← copy this to EOT/model/")
print("  training_curves.png ← training graph")
print("  confusion_matrix.png← confusion matrix")
print("=" * 50)