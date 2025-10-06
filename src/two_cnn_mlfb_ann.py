# -*- coding: utf-8 -*-
"""
2CNN-MLFB-ANN for beehive image classification.
"""
from pathlib import Path
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ------------------------- config ------------------------------------
IMAGE_PATH = Path(os.getenv("IMAGE_PATH", REPO_ROOT / "data")).resolve()
SEED = 42
IMAGE_WIDTH = 200
IMAGE_HEIGHT = 200
BATCH_SIZE = 32
wd = 1e-4

FIG_DIR = (REPO_ROOT / "figures"); FIG_DIR.mkdir(parents=True, exist_ok=True)
RES_DIR = (REPO_ROOT / "results"); RES_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------- data discovery ----------------------------------------------------
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
# 70/10/20 (train/val/test) as used for the referenced “second run”
train_df, temp_df = train_test_split(
    df, test_size=0.30, stratify=df["health"], random_state=SEED
)
test_df, val_df = train_test_split(
    temp_df, test_size=1/3, stratify=temp_df["health"], random_state=SEED
)

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

# ----------------------- model ----------------------------
input_layer = tf.keras.Input(shape=(IMAGE_WIDTH, IMAGE_HEIGHT, 3))

# First CNN branch (3×3 filters)
x1 = layers.Conv2D(32, (3, 3), padding='same', use_bias=False,
                   kernel_regularizer=regularizers.l2(wd))(input_layer)
x1 = layers.BatchNormalization()(x1); x1 = layers.ReLU()(x1)
x1 = layers.MaxPooling2D((2, 2))(x1)
x1 = layers.Conv2D(64, (3, 3), padding='same', use_bias=False,
                   kernel_regularizer=regularizers.l2(wd))(x1)
x1 = layers.BatchNormalization()(x1); x1 = layers.ReLU()(x1)
x1 = layers.MaxPooling2D((2, 2))(x1)
x1 = layers.Conv2D(128, (3, 3), padding='same', use_bias=False,
                   kernel_regularizer=regularizers.l2(wd))(x1)
x1 = layers.BatchNormalization()(x1); x1 = layers.ReLU()(x1)
x1 = layers.GlobalAveragePooling2D()(x1)

# Second CNN branch (5×5 filters)
x2 = layers.Conv2D(32, (5, 5), padding='same', use_bias=False,
                   kernel_regularizer=regularizers.l2(wd))(input_layer)
x2 = layers.BatchNormalization()(x2); x2 = layers.ReLU()(x2)
x2 = layers.MaxPooling2D((2, 2))(x2)
x2 = layers.Conv2D(64, (5, 5), padding='same', use_bias=False,
                   kernel_regularizer=regularizers.l2(wd))(x2)
x2 = layers.BatchNormalization()(x2); x2 = layers.ReLU()(x2)
x2 = layers.MaxPooling2D((2, 2))(x2)
x2 = layers.Conv2D(128, (5, 5), padding='same', use_bias=False,
                   kernel_regularizer=regularizers.l2(wd))(x2)
x2 = layers.BatchNormalization()(x2); x2 = layers.ReLU()(x2)
x2 = layers.GlobalAveragePooling2D()(x2)

# Combine two feature vectors (128 + 128 = 256)
combined_features = layers.concatenate([x1, x2])

# MLFB-ANN classification head
dense1 = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(wd))(combined_features)
dense1 = layers.Dropout(0.2)(dense1)
feedback_input = layers.concatenate([combined_features, dense1])
dense2 = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(wd))(feedback_input)
dense2 = layers.Dropout(0.2)(dense2)
output_layer = layers.Dense(num_classes, activation='softmax')(dense2)

model = models.Model(inputs=input_layer, outputs=output_layer)
model.summary()

# -------------------- optimizer & loss --------------------
try:
    from tensorflow.keras.optimizers import AdamW
except Exception:
    from tensorflow.keras.optimizers.experimental import AdamW

opt = AdamW(learning_rate=1e-3, weight_decay=1e-4, clipnorm=1.0)
loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05)

model.compile(optimizer=opt, loss=loss, metrics=['accuracy'])

# ------------------------ callbacks -----------------------
cbs = [
    tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=7, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1),
    tf.keras.callbacks.ModelCheckpoint("best.h5", monitor="val_loss", save_best_only=True, save_weights_only=True),
]

# ------------------------ training ------------------------
history = model.fit(train_flow, epochs=100, validation_data=val_flow, callbacks=cbs, verbose=1)

# ------------------------ evaluation ----------------------
test_loss, test_acc = model.evaluate(test_flow, verbose=1)
print(f"Test Accuracy: {test_acc*100:.2f}%")
print(f"Test Loss: {test_loss*100:.2f}%")

y_pred_prob = model.predict(test_flow)
y_pred = np.argmax(y_pred_prob, axis=1)
y_true = test_flow.classes

macro_f1 = f1_score(y_true, y_pred, average='macro')
print("Macro-F1:", macro_f1)

report_text = classification_report(y_true, y_pred, target_names=classes)
print("Classification Report:\n", report_text)

# -------------------- visualizations --------------------------------------
# 1) Training curves
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
plt.savefig(FIG_DIR / 'training_curves_2CNN_MLFB_ANN.pdf', format='pdf')
plt.close()

# 2) Confusion matrix
cm = confusion_matrix(y_true, y_pred)
fig, ax = plt.subplots(figsize=(6, 5))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
disp.plot(ax=ax, cmap=plt.cm.Oranges, values_format='d')
plt.title('Confusion Matrix'); plt.xlabel('Predicted Class'); plt.ylabel('True Class'); plt.tight_layout()
plt.savefig(FIG_DIR / 'confusion_matrix_2CNN_MLFB_ANN.pdf', format='pdf')
plt.close()

# 3) Per-class metrics bar chart
report_dict = classification_report(y_true, y_pred, target_names=classes, output_dict=True)
precisions = [report_dict[cls]['precision'] for cls in classes]
recalls = [report_dict[cls]['recall'] for cls in classes]
f1_scores = [report_dict[cls]['f1-score'] for cls in classes]

x = np.arange(len(classes))
width = 0.25
plt.figure(figsize=(8, 5))
plt.bar(x - width, precisions, width, label='Precision')
plt.bar(x,         recalls,   width, label='Recall')
plt.bar(x + width, f1_scores, width, label='F1-score')
plt.xticks(x, classes, rotation=45)
plt.ylim(0, 1.0)
plt.title('Per-Class Performance Metrics')
plt.xlabel('Class'); plt.ylabel('Score'); plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig(FIG_DIR / 'per_class_metrics_2CNN_MLFB_ANN.pdf', format='pdf')
plt.close()

with open(RES_DIR / 'two_cnn_mlfb_ann_classification_report.txt', 'w', encoding='utf-8') as f:
    f.write(f"Test Accuracy: {test_acc*100:.2f}%\n")
    f.write(f"Test Loss: {test_loss*100:.2f}%\n")
    f.write(f"Macro-F1: {macro_f1:.6f}\n\n")
    f.write(report_text)
