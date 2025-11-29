import tensorflow as tf

print("🔍 Loading data...")
train = tf.keras.preprocessing.image_dataset_from_directory(
    "../data/split/train",
    image_size=(224, 224),
    batch_size=16,
    label_mode="categorical"
)
val = tf.keras.preprocessing.image_dataset_from_directory(
    "../data/split/val",
    image_size=(224, 224),
    batch_size=16,
    label_mode="categorical"
)

print("✅ Datasets ready:", len(train), "train batches,", len(val), "val batches")

base = tf.keras.applications.MobileNetV2(
    input_shape=(224,224,3),
    include_top=False,
    weights="imagenet"
)
base.trainable = False

model = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1./127.5, offset=-1),
    base,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(6, activation="softmax")
])
model.compile(optimizer="adam",
              loss="categorical_crossentropy",
              metrics=["accuracy"])

print("🚀 Starting debug training (3 epochs)...")
history = model.fit(train, validation_data=val, epochs=3, verbose=1)
print("✅ Done!")
