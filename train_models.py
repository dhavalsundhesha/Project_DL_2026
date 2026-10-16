"""
Train and compare four MNIST classifiers:
1. MLP (fully connected neural network)
2. LeNet-5-style CNN
3. Custom CNN
4. Small ResNet-style CNN

Run: python train_models.py
Optional: EPOCHS=5 python train_models.py
Metrics are generated from actual training; no performance values are hard-coded.
"""
import os, json, time, csv
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)
EPOCHS = int(os.environ.get("EPOCHS", "5"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "128"))

def load_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    x_train = x_train[..., None]
    x_test = x_test[..., None]
    return (x_train, y_train), (x_test, y_test)

def mlp_model():
    inp = tf.keras.Input((28, 28, 1), name="image")
    x = tf.keras.layers.Flatten()(inp)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    out = tf.keras.layers.Dense(10, activation="softmax")(x)
    return tf.keras.Model(inp, out, name="MLP")

def lenet_model():
    inp = tf.keras.Input((28, 28, 1), name="image")
    x = tf.keras.layers.Conv2D(6, 5, padding="same", activation="tanh", name="lenet_conv1")(inp)
    x = tf.keras.layers.AveragePooling2D(2)(x)
    x = tf.keras.layers.Conv2D(16, 5, activation="tanh", name="lenet_conv2")(x)
    x = tf.keras.layers.AveragePooling2D(2)(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(120, activation="tanh")(x)
    x = tf.keras.layers.Dense(84, activation="tanh")(x)
    out = tf.keras.layers.Dense(10, activation="softmax")(x)
    return tf.keras.Model(inp, out, name="LeNet5_style")

def custom_cnn_model():
    inp = tf.keras.Input((28, 28, 1), name="image")
    x = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu", name="custom_conv1")(inp)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu", name="custom_conv2")(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu", name="custom_conv3")(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    out = tf.keras.layers.Dense(10, activation="softmax")(x)
    return tf.keras.Model(inp, out, name="Custom_CNN")

def residual_block(x, filters, name):
    shortcut = x
    x = tf.keras.layers.Conv2D(filters, 3, padding="same", use_bias=False, name=f"{name}_conv1")(x)
    x = tf.keras.layers.BatchNormalization(name=f"{name}_bn1")(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Conv2D(filters, 3, padding="same", use_bias=False, name=f"{name}_conv2")(x)
    x = tf.keras.layers.BatchNormalization(name=f"{name}_bn2")(x)
    if shortcut.shape[-1] != filters:
        shortcut = tf.keras.layers.Conv2D(filters, 1, padding="same", use_bias=False, name=f"{name}_projection")(shortcut)
        shortcut = tf.keras.layers.BatchNormalization(name=f"{name}_projection_bn")(shortcut)
    x = tf.keras.layers.Add()([x, shortcut])
    return tf.keras.layers.Activation("relu")(x)

def resnet_model():
    inp = tf.keras.Input((28, 28, 1), name="image")
    x = tf.keras.layers.Conv2D(32, 3, padding="same", use_bias=False, name="resnet_stem_conv")(inp)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = residual_block(x, 32, "res_block1")
    x = tf.keras.layers.MaxPooling2D()(x)
    x = residual_block(x, 64, "res_block2")
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    out = tf.keras.layers.Dense(10, activation="softmax")(x)
    return tf.keras.Model(inp, out, name="Small_ResNet")

MODELS = {
    "MLP": mlp_model,
    "LeNet5": lenet_model,
    "CustomCNN": custom_cnn_model,
    "ResNet": resnet_model,
}

def main():
    (x_train, y_train), (x_test, y_test) = load_data()
    results = []
    for key, factory in MODELS.items():
        print(f"\n{'='*60}\nTraining {key} for {EPOCHS} epochs")
        model = factory()
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                      loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        start = time.time()
        history = model.fit(
            x_train, y_train,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_split=0.1,
            verbose=2
        )
        train_seconds = time.time() - start
        probs = model.predict(x_test, batch_size=BATCH_SIZE, verbose=0)
        pred = np.argmax(probs, axis=1)
        test_loss, test_acc = model.evaluate(x_test, y_test, batch_size=BATCH_SIZE, verbose=0)
        precision = precision_score(y_test, pred, average="weighted", zero_division=0)
        recall = recall_score(y_test, pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, pred, average="weighted", zero_division=0)
        model_path = os.path.join("models", f"{key}.keras")
        model.save(model_path)
        cm = confusion_matrix(y_test, pred)
        np.savetxt(os.path.join("results", f"{key}_confusion_matrix.csv"), cm, delimiter=",", fmt="%d")
        with open(os.path.join("results", f"{key}_classification_report.txt"), "w") as f:
            f.write(classification_report(y_test, pred, digits=4))
        with open(os.path.join("results", f"{key}_history.json"), "w") as f:
            json.dump({k: [float(v) for v in vals] for k, vals in history.history.items()}, f, indent=2)
        plt.figure(figsize=(8, 4))
        plt.plot(history.history["accuracy"], label="Training accuracy")
        plt.plot(history.history["val_accuracy"], label="Validation accuracy")
        plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.title(f"{key} accuracy")
        plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join("results", f"{key}_accuracy.png"), dpi=150); plt.close()
        plt.figure(figsize=(8, 4))
        plt.plot(history.history["loss"], label="Training loss")
        plt.plot(history.history["val_loss"], label="Validation loss")
        plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title(f"{key} loss")
        plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join("results", f"{key}_loss.png"), dpi=150); plt.close()
        row = {
            "model": key, "test_accuracy": float(test_acc),
            "weighted_precision": float(precision), "weighted_recall": float(recall),
            "weighted_f1": float(f1), "test_loss": float(test_loss),
            "training_time_seconds": round(train_seconds, 2), "epochs": EPOCHS,
            "batch_size": BATCH_SIZE, "learning_rate": 0.001,
            "optimizer": "Adam", "loss_function": "sparse_categorical_crossentropy"
        }
        results.append(row)
        print(f"{key}: test accuracy={test_acc:.4f}, precision={precision:.4f}, "
              f"recall={recall:.4f}, F1={f1:.4f}, training time={train_seconds:.1f}s")
    with open(os.path.join("results", "model_comparison.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader(); writer.writerows(results)
    with open(os.path.join("results", "model_comparison.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("\nAll models trained. Comparison saved to results/model_comparison.csv")

if __name__ == "__main__":
    main()
