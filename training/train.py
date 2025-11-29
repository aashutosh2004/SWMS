import os
import json
import random
import shutil
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

DATA_DIR = os.path.join('..', 'data')
RAW_DIR = os.path.join(DATA_DIR, 'raw')
SPLIT_DIR = os.path.join(DATA_DIR, 'split')
LABELS = os.path.join(DATA_DIR, 'labels.json')
MODEL_DIR = os.path.join('..', 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'waste_mobilenetv2.h5')

IMG_SIZE = (224, 224)
BATCH_SIZE = 16
CLASSES = ['cardboard','glass','metal','paper','plastic','trash']

def make_split():
    os.makedirs(SPLIT_DIR, exist_ok=True)
    if len(os.listdir(SPLIT_DIR)) > 0:
        print("Split already exists")
        return
    tr, val, ts = 0.7, 0.15, 0.15
    for c in CLASSES:
        src = os.path.join(RAW_DIR, c)
        imgs = [x for x in os.listdir(src) if x.lower().endswith(('jpg','jpeg','png'))]
        random.shuffle(imgs)
        n = len(imgs)
        trn, vl = int(n*tr), int(n*val)
        for name, arr in {'train': imgs[:trn], 'val': imgs[trn:trn+vl], 'test': imgs[trn+vl:]}.items():
            dst = os.path.join(SPLIT_DIR, name, c)
            os.makedirs(dst, exist_ok=True)
            for f in arr:
                shutil.copy(os.path.join(src, f), os.path.join(dst, f))
    print("✅ Dataset split created")

def load_data():
    train = tf.keras.preprocessing.image_dataset_from_directory(
        os.path.join(SPLIT_DIR, 'train'),
        image_size=IMG_SIZE, batch_size=BATCH_SIZE, label_mode='categorical')
    val = tf.keras.preprocessing.image_dataset_from_directory(
        os.path.join(SPLIT_DIR, 'val'),
        image_size=IMG_SIZE, batch_size=BATCH_SIZE, label_mode='categorical')
    test = tf.keras.preprocessing.image_dataset_from_directory(
        os.path.join(SPLIT_DIR, 'test'),
        image_size=IMG_SIZE, batch_size=BATCH_SIZE, label_mode='categorical', shuffle=False)
    with open(LABELS, 'w') as f:
        json.dump(train.class_names, f)
    return train.prefetch(tf.data.AUTOTUNE), val.prefetch(tf.data.AUTOTUNE), test.prefetch(tf.data.AUTOTUNE)

def build_model(num_classes):
    base = tf.keras.applications.MobileNetV2(input_shape=IMG_SIZE + (3,), include_top=False, weights='imagenet')
    base.trainable = False
    model = models.Sequential([
        tf.keras.layers.Rescaling(1./127.5, offset=-1),
        base,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model, base

def main():
    make_split()
    train, val, test = load_data()

    model, base = build_model(6)
    print("🚀 Starting Training...")
    history = model.fit(train, validation_data=val, epochs=10, verbose=1)

    print("🔧 Fine-tuning MobileNetV2 (unfreezing last layers)...")
    base.trainable = True
    for layer in base.layers[:-60]:
        layer.trainable = False

    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    ft_history = model.fit(train, validation_data=val, epochs=5, verbose=1)

    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save(MODEL_PATH)
    print("✅ Model saved:", MODEL_PATH)

    y_true, y_pred = [], []
    for x, y in test:
        p = model.predict(x, verbose=0)
        y_true.extend(np.argmax(y.numpy(), axis=1))
        y_pred.extend(np.argmax(p, axis=1))

    print(classification_report(y_true, y_pred, target_names=CLASSES))
    cm = confusion_matrix(y_true, y_pred)
    plt.imshow(cm, cmap='Blues')
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.xticks(np.arange(len(CLASSES)), CLASSES, rotation=45)
    plt.yticks(np.arange(len(CLASSES)), CLASSES)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
