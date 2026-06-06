import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras import layers
from tensorflow.keras import mixed_precision
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

mixed_precision.set_global_policy("mixed_float16")
print("Mixed precision policy:", mixed_precision.global_policy())

SELECTED_CLASSES = [
    "pizza", "hamburger", "french_fries", "ice_cream", "chocolate_cake",
    "sushi", "ramen", "fried_rice", "omelette", "pancakes",
    "hot_dog", "grilled_salmon", "caesar_salad", "donuts", "dumplings",
]
(train_data, test_data), ds_info = tfds.load(
    "food101",
    split=["train", "validation"],
    shuffle_files=True,
    as_supervised=True,
    with_info=True
)
class_names = ds_info.features["label"].names
selected_ids = tf.constant(
    [class_names.index(name) for name in SELECTED_CLASSES],
    dtype=tf.int64
)
def filter_classes(image, label):
    return tf.reduce_any(tf.equal(label, selected_ids))

IMG_SIZE = 224
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE
def preprocess_image(image, label):
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    image = tf.cast(image, tf.float32)
    image = tf.keras.applications.efficientnet.preprocess_input(image)
    return image, label

train_data = (
    train_data
    .filter(filter_classes)
    .cache("/tmp/train_cache")
    .map(preprocess_image, num_parallel_calls=AUTOTUNE)
    .shuffle(1000)
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)
test_data = (
    test_data
    .filter(filter_classes)
    .cache("/tmp/test_cache")
    .map(preprocess_image, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)

table = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(
        keys=selected_ids,
        values=tf.range(len(SELECTED_CLASSES), dtype=tf.int64)
    ),
    default_value=-1
)
def remap_label(image, label):
    label = table.lookup(label)
    return image, label

train_data = train_data.map(remap_label, num_parallel_calls=AUTOTUNE)
test_data = test_data.map(remap_label, num_parallel_calls=AUTOTUNE)

def to_onehot(image, label):
    return image, tf.one_hot(label, len(SELECTED_CLASSES))
test_data_onehot = test_data.map(to_onehot, num_parallel_calls=AUTOTUNE)

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.1),
])
def apply_mixup(images, labels, alpha=0.2):
    batch_size = tf.shape(images)[0]
    lam = tf.random.uniform([batch_size, 1, 1, 1], 0.0, 1.0)
    lam = tf.maximum(lam, 1.0 - lam)
    images_mix = lam * images + (1 - lam) * tf.reverse(images, [0])
    labels_onehot = tf.one_hot(labels, len(SELECTED_CLASSES))
    lam_2d = tf.reshape(lam, [batch_size, 1])
    labels_mix = (
        lam_2d * labels_onehot
        + (1 - lam_2d) * tf.reverse(labels_onehot, [0])
    )
    return images_mix, labels_mix
def mixup_dataset(dataset, prob=0.5):
    def _mixup_batch(images, labels):
        r = tf.random.uniform([])
        return tf.cond(
            r < prob,
            lambda: apply_mixup(images, labels),
            lambda: (images, tf.one_hot(labels, len(SELECTED_CLASSES)))
        )
    return dataset.map(_mixup_batch, num_parallel_calls=AUTOTUNE)
train_data_mix = mixup_dataset(train_data, prob=0.5)

NUM_TRAIN_IMAGES = 750 * len(SELECTED_CLASSES)
steps_per_epoch  = NUM_TRAIN_IMAGES // BATCH_SIZE
train_data_mix = mixup_dataset(train_data, prob=0.5).repeat()
class_weight = {i: 1.0 for i in range(len(SELECTED_CLASSES))}
print("Class weights:", class_weight)

base_model = tf.keras.applications.EfficientNetB0(include_top=False)
base_model.trainable = False
inputs = layers.Input(shape=(224, 224, 3))
x = data_augmentation(inputs)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(
    len(SELECTED_CLASSES),
    activation="softmax",
    dtype=tf.float32
)(x)
model = tf.keras.Model(inputs, outputs)

loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)
initial_lr_1  = 1e-3
decay_steps_1 = steps_per_epoch * 5
cosine_decay_1 = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=initial_lr_1,
    decay_steps=decay_steps_1,
    alpha=0.01
)
optimizer_1 = tf.keras.optimizers.AdamW(
    learning_rate=cosine_decay_1, weight_decay=1e-4
)
model.compile(loss=loss_fn, optimizer=optimizer_1, metrics=["accuracy"])

early_stopping_1 = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=3, restore_best_weights=True
)
reduce_lr_1 = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss", factor=0.2, patience=2, verbose=1
)
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    "best_model.keras", save_best_only=True
)

history_1 = model.fit(
    train_data_mix,
    epochs=5,
    steps_per_epoch=steps_per_epoch,
    validation_data=test_data_onehot,
    callbacks=[early_stopping_1, reduce_lr_1, checkpoint],
)

base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False
initial_lr_2 = 1e-5
decay_steps_2 = steps_per_epoch * 20
cosine_decay_2 = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=initial_lr_2,
    decay_steps=decay_steps_2,
    alpha=0.1
)
optimizer_2 = tf.keras.optimizers.AdamW(learning_rate=cosine_decay_2, weight_decay=1e-4)
model.compile(loss=loss_fn, optimizer=optimizer_2, metrics=["accuracy"])
early_stopping_2 = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=5, restore_best_weights=True
)
reduce_lr_2 = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss", factor=0.2, patience=2, verbose=1
)
history_2 = model.fit(
    train_data_mix,
    epochs=20,
    steps_per_epoch=steps_per_epoch,
    validation_data=test_data_onehot,
    callbacks=[early_stopping_2, reduce_lr_2, checkpoint],
)

acc      = history_1.history["accuracy"]     + history_2.history["accuracy"]
val_acc  = history_1.history["val_accuracy"] + history_2.history["val_accuracy"]
loss     = history_1.history["loss"]         + history_2.history["loss"]
val_loss = history_1.history["val_loss"]     + history_2.history["val_loss"]
fine_tune_start = len(history_1.history["accuracy"])
epochs = range(1, len(acc) + 1)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
for ax, train, val, title in [
    (ax1, acc,  val_acc,  "Accuracy"),
    (ax2, loss, val_loss, "Loss")
]:
    ax.plot(epochs, train, label="Train")
    ax.plot(epochs, val,   label="Validation")
    ax.axvline(fine_tune_start, linestyle="--", color="gray", label="Fine-tune start")
    ax.set_title(title); ax.set_xlabel("Epoch"); ax.legend()
plt.tight_layout()
plt.savefig("learning_curve.png", dpi=150)
plt.show()
y_true, y_pred = [], []
for images, labels in test_data:
    preds     = model.predict(images, verbose=0)
    y_pred.extend(np.argmax(preds, axis=1))
    y_true.extend(labels.numpy())
y_true     = np.array(y_true)
y_pred     = np.array(y_pred)
misclassified = np.where(np.array(y_pred) != np.array(y_true))[0]
cm      = confusion_matrix(y_true, y_pred)
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
fig, axes = plt.subplots(1, 2, figsize=(22, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=SELECTED_CLASSES, yticklabels=SELECTED_CLASSES, ax=axes[0])
axes[0].set_title("Confusion Matrix (counts)")
axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("True")

sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=SELECTED_CLASSES, yticklabels=SELECTED_CLASSES, ax=axes[1])
axes[1].set_title("Confusion Matrix (normalized)")
axes[1].set_xlabel("Predicted"); axes[1].set_ylabel("True")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.show()

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=SELECTED_CLASSES, digits=4))
f1_scores  = f1_score(y_true, y_pred, average=None)
sorted_idx = np.argsort(f1_scores)
plt.figure(figsize=(10, 6))
plt.barh(
    [SELECTED_CLASSES[i] for i in sorted_idx],
    f1_scores[sorted_idx],
    color=["#d9534f" if f1_scores[i] < 0.8 else "#5cb85c" for i in sorted_idx]
)
plt.axvline(f1_scores.mean(), linestyle="--", color="gray",
            label=f"Mean F1 = {f1_scores.mean():.3f}")
plt.xlabel("F1-score"); plt.title("Per-class F1-score")
plt.legend(); plt.tight_layout()
plt.savefig("f1_per_class.png", dpi=150)
plt.show()
misclassified = np.where(y_pred != y_true)[0]
print(f"Số ảnh sai: {len(misclassified)} / {len(y_true)}")
if len(misclassified) > 0:
    plt.figure(figsize=(15, 5))
    for i, idx in enumerate(misclassified[:5]):
        plt.subplot(1, 5, i + 1)
        img = (all_images[idx] - all_images[idx].min()) / \
              (all_images[idx].max() - all_images[idx].min())
        plt.imshow(img.astype(np.float32))
        plt.title(f"True: {SELECTED_CLASSES[y_true[idx]]}\n"
                  f"Pred: {SELECTED_CLASSES[y_pred[idx]]}")
        plt.axis("off")
    plt.tight_layout()
    plt.savefig("misclassified.png", dpi=150)
    plt.show()