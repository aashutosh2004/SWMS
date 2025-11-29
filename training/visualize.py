import os
import json
import matplotlib.pyplot as plt
import tensorflow as tf

DATA_DIR = os.path.join('..', 'data')
SPLIT_DIR = os.path.join(DATA_DIR, 'split')
LABELS_JSON = os.path.join(DATA_DIR, 'labels.json')

with open(LABELS_JSON) as f:
    classes = json.load(f)

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    os.path.join(SPLIT_DIR, 'train'),
    image_size=(224,224),
    batch_size=32,
    label_mode='categorical'
)

# approx class distribution
counts = {cls: 0 for cls in classes}

for images, labels in train_ds.take(200):   # 200 batches enough sample
    for row in labels.numpy():
        idx = int(tf.argmax(row).numpy())
        counts[classes[idx]] += 1

plt.figure(figsize=(7,4))
plt.bar(counts.keys(), counts.values())
plt.title("Class Distribution (Train)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# show sample images
for images, labels in train_ds.take(1):
    plt.figure(figsize=(8,8))
    for i in range(16):
        ax = plt.subplot(4,4,i+1)
        plt.imshow(images[i].numpy().astype("uint8"))
        idx = int(tf.argmax(labels[i]).numpy())
        plt.title(classes[idx])
        plt.axis("off")
    plt.tight_layout()
    plt.show()
