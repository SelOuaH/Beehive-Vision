# -*- coding: utf-8 -*-
"""
Baseline CNN for beehive image classification.
"""
from pathlib import Path
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay, f1_score
)

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping, ModelCheckpoint

# ------------------------- config -------------------------
IMAGE_PATH = Path(os.getenv("IMAGE_PATH", REPO_ROOT / "data")).resolve()
SEED = 42
IMAGE_WIDTH = 200
IMAGE_HEIGHT = 200
CONV_2D_DIM_1 = 32
CONV_2D_DIM_2 = 64
MAX_POOL_DIM = 2
IMAGE_CHANNELS = 3
KERNEL_SIZE = 3
BATCH_SIZE = 32
PATIENCE = 5
VERBOSE = 1

FIG_DIR = Path("./figures"); FIG_DIR.mkdir(parents=True, exist_ok=True)
RES_DIR = Path("./results"); RES_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------- data discovery --------------------
valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff"}
files = [p for p in IMAGE_PATH.rglob("*") if p.suffix.lower() in valid_exts and p.is_file()]

df = pd.DataFrame({
    "file":   [p.relative_to(IMAGE_PATH).as_posix() for p in files],
    "health": [p.parent.name for p in files],
})
df["health"] = pd.Categorical(df["health"])
classes = sorted(df["health"].cat.categories.tolist())
num_classes = len(classes)

# ------------------- stratified splits --------------------
# Second run setting: 70/10/20 (train/val/test)
train_df, temp_df = train_test_split(
    df, test_size=0.30, stratify=df["health"], random_state=SEED
)
test_df, val_df = train_test_split(
    temp_df, test_size=1/3, stratify=temp_df["health"], random_state=SEED
)

print("Counts:", len(train_df), len(val_df), len(test_df),
      "ratios:", round(len(train_df)/len(df),3), round(len(val_df)/len(df),3), round(len(test_df)/len(df),3))

# -------------------- generators --------------------------
train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10, width_shift_range=0.05, height_shift_range=0.05,
    shear_range=0.05, zoom_range=0.05, horizontal_flip=True, fill_mode="nearest"
)
eval_gen = ImageDataGenerator(rescale=1./255)

train_flow = train_gen.flow_from_dataframe(
    dataframe=train_df,
    directory=str(IMAGE_PATH),
    x_col='file', y_col='health',
    target_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
    class_mode="categorical",
    classes=classes,
    shuffle=True,
    batch_size=BATCH_SIZE,
    seed=SEED
)

val_flow = eval_gen.flow_from_dataframe(
    dataframe=val_df,
    directory=str(IMAGE_PATH),
    x_col='file', y_col='health',
    target_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
    class_mode="categorical",
    classes=classes,
    shuffle=False,
    batch_size=BATCH_SIZE
)

test_flow = eval_gen.flow_from_dataframe(
    dataframe=test_df,
    directory=str(IMAGE_PATH),
    x_col='file', y_col='health',
    target_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
    class_mode="categorical",
    classes=classes,
    shuffle=False,
    batch_size=BATCH_SIZE
)

print(train_flow.class_indices)

# ----------------------- model ----------------------------
model = Sequential()
model.add(Conv2D(CONV_2D_DIM_1, kernel_size=3, input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS),
                  activation='relu', padding='same'))
model.add(MaxPool2D(MAX_POOL_DIM))
model.add(Dropout(0.2))
model.add(Conv2D(CONV_2D_DIM_2, kernel_size=KERNEL_SIZE, activation='relu', padding='same'))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(num_classes, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

annealer = LearningRateScheduler(lambda x: 1e-3 * 0.995 ** (x+100))
earlystopper = EarlyStopping(monitor='loss', patience=PATIENCE, verbose=VERBOSE)
checkpointer = ModelCheckpoint('best_model_3.h5',
                                monitor='val_acc',   
                                verbose=VERBOSE,
                                save_best_only=True,
                                save_weights_only=True)

# ---------------------- training --------------------------
history = model.fit(
    train_flow, epochs=100, validation_data=val_flow,
    callbacks=[earlystopper, checkpointer, annealer]
)

# -------------------- evaluation --------------------------
test_loss, test_acc = model.evaluate(test_flow, verbose=VERBOSE)
print(f"Test Accuracy: {test_acc*100:.2f}%")
print(f"Test Loss: {test_loss*100:.2f}%")

y_pred_prob = model.predict(test_flow)
y_pred = np.argmax(y_pred_prob, axis=1)
y_true = test_flow.classes
class_labels = classes

macro_f1 = f1_score(y_true, y_pred, average='macro')
print("Macro-F1:", macro_f1)

report_text = classification_report(y_true, y_pred, target_names=class_labels)
print("Classification Report:\n", report_text)

# -------------------- visualizations ----------------------
epochs_range = range(len(history.history['loss']))
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, history.history['accuracy'], label='Train Accuracy')
plt.plot(epochs_range, history.history['val_accuracy'], label='Val Accuracy')
plt.title('Accuracy over Epochs'); plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend()
plt.subplot(1, 2, 2)
plt.plot(epochs_range, history.history['loss'], label='Train Loss')
plt.plot(epochs_range, history.history['val_loss'], label='Val Loss')
plt.title('Loss over Epochs'); plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()
plt.tight_layout()
plt.savefig(FIG_DIR / 'training_curvesBaselineCNN.pdf', format='pdf')
plt.close()

cm = confusion_matrix(y_true, y_pred)
fig, ax = plt.subplots(figsize=(6, 5))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
disp.plot(ax=ax, cmap=plt.cm.Oranges, values_format='d')
plt.title('Confusion Matrix'); plt.xlabel('Predicted Class'); plt.ylabel('True Class'); plt.tight_layout()
plt.savefig(FIG_DIR / 'confusion_matrixBaselineCNN.pdf', format='pdf')
plt.close()

report_dict = classification_report(y_true, y_pred, target_names=class_labels, output_dict=True)
precisions = [report_dict[cls]['precision'] for cls in class_labels]
recalls = [report_dict[cls]['recall'] for cls in class_labels]
f1_scores = [report_dict[cls]['f1-score'] for cls in class_labels]
x = np.arange(len(class_labels))
width = 0.25
plt.figure(figsize=(8, 5))
plt.bar(x - width, precisions, width, label='Precision')
plt.bar(x, recalls, width, label='Recall')
plt.bar(x + width, f1_scores, width, label='F1-score')
plt.xticks(x, class_labels, rotation=45); plt.ylim(0, 1.0)
plt.title('Per-Class Performance Metrics'); plt.xlabel('Class'); plt.ylabel('Score'); plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig(FIG_DIR / 'per_class_metricsBaselineCNN.pdf', format='pdf')
plt.close()

# ---------------------- save report -----------------------
with open(RES_DIR / 'baseline_classification_report.txt', 'w', encoding='utf-8') as f:
    f.write(f"Test Accuracy: {test_acc*100:.2f}%\n")
    f.write(f"Test Loss: {test_loss*100:.2f}%\n")
    f.write(f"Macro-F1: {macro_f1:.6f}\n\n")
    f.write(report_text)
