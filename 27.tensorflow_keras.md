# TensorFlow and Keras

> Notes written in a GeeksforGeeks-style tutorial format — simple definitions first, then deeper detail, tables, and code examples. Builds on `neural_network.md`.

---

## What is TensorFlow?

**TensorFlow** is an open-source **deep learning framework** developed by Google, used to build, train, and deploy machine learning and neural network models. It provides a complete ecosystem — from writing a model, to training it on CPUs/GPUs/TPUs, to deploying it on servers, mobile devices, browsers, and embedded devices.

The name comes from its core data structure — the **Tensor** — and the fact that computations are represented as a **flow** of tensors through a graph of operations.

**Tensor:** simply a multi-dimensional array with a uniform data type (dtype)
- tensors "flow" through a computational graph, changing shape and value as they pass through neural network layers
- tensors "flow" through a computational graph, changing shape and value as they pass through different mathematical operations and neural network layers

**Core Properties of a TensorEvery tensor** 
1. **Dimensions:** The number of axes the tensor has. 
2. **Shape:** The size or length of each axis.
3. **Data Type (dtype):** The type of data stored inside the tensor. All elements inside a single tensor must share the same data type

**Key highlights:**
- Handles the heavy numerical computation (matrix multiplications, gradients, etc.) efficiently using optimized C++ backends and GPU/TPU acceleration.
- Automatically computes gradients using **Automatic Differentiation**, which is what makes training via backpropagation practical without hand-deriving derivatives.
- Provides a huge ecosystem: TensorFlow Lite (mobile), TensorFlow.js (browser), TensorFlow Serving (production deployment), TensorFlow Extended/TFX (ML pipelines).

## What is Keras?

**Keras** is a **high-level neural network API** that provides a simple, user-friendly interface for building and training deep learning models. Since **TensorFlow 2.0**, Keras has been fully integrated into TensorFlow as its official high-level API — accessible via `tensorflow.keras`.

In simple words: **TensorFlow** is the powerful low-level engine that does the heavy lifting (tensor operations, gradients, hardware acceleration), while **Keras** is the easy-to-use interface sitting on top of it that lets you define a neural network in just a few lines of readable code.

```
┌───────────────────────────────┐
│   Keras (tf.keras)            │  ← High-level API: Sequential, Model, layers.Dense...
├───────────────────────────────┤
│   TensorFlow Core              │  ← Tensors, Automatic Differentiation, tf.function
├───────────────────────────────┤
│   Hardware (CPU / GPU / TPU)  │
└───────────────────────────────┘
```

---

## TensorFlow vs Keras — Quick Comparison

| | TensorFlow | Keras (tf.keras) |
|---|---|---|
| **Level** | Low-level + high-level | High-level only |
| **Ease of use** | More verbose, more control | Simple, beginner-friendly |
| **Flexibility** | Full control over every operation | Covers most use cases; drop down to TF for custom needs |
| **Relationship** | The full framework | Built **into** TensorFlow (`tf.keras`) |
| **Best for** | Custom research, low-level control, production pipelines | Rapid prototyping, standard architectures |

**Note:** You almost never "choose between" them in modern TensorFlow — you use **Keras (`tf.keras`) as the day-to-day API** and drop down to raw TensorFlow operations (`tf.GradientTape`, custom training loops) only when you need finer control.

---

## What is a Tensor?

A **Tensor** is the fundamental data structure in TensorFlow — a multi-dimensional array, similar to a NumPy array, but with the added ability to run on GPUs/TPUs and track gradients automatically.

| Rank | Name | Example |
|---|---|---|
| 0 | Scalar | `5` |
| 1 | Vector | `[1, 2, 3]` |
| 2 | Matrix | `[[1, 2], [3, 4]]` |
| 3+ | N-dimensional Tensor | A batch of RGB images: `(batch, height, width, channels)` |

```python
import tensorflow as tf

scalar = tf.constant(5)
vector = tf.constant([1, 2, 3])
matrix = tf.constant([[1, 2], [3, 4]])

print(matrix.shape)   # (2, 2)
print(matrix.dtype)   # <dtype: 'int32'>
```

---

## Core TensorFlow Concepts

### 1. Eager Execution

By default, TensorFlow 2.x runs in **Eager Execution** mode — operations execute immediately and return concrete values, just like normal Python/NumPy code. (Older TensorFlow 1.x required building a static computation graph first and running it inside a `Session` — this is no longer the default.)

```python
a = tf.constant(2)
b = tf.constant(3)
print(a + b)   # tf.Tensor(5, shape=(), dtype=int32) — computed immediately
```

### 2. `tf.function` — Graph Mode

Decorating a Python function with `@tf.function` compiles it into a fast, optimized **static computation graph** (using **AutoGraph**), which can significantly speed up training and enables deployment outside Python. This gives the best of both worlds: easy-to-debug eager code during development, and graph-mode speed in production.

```python
@tf.function
def compute(x, y):
    return x * y + 1

result = compute(tf.constant(2), tf.constant(3))
```

### 3. Automatic Differentiation (`tf.GradientTape`)

TensorFlow automatically computes gradients — the backbone of backpropagation — using `tf.GradientTape`, which "records" operations performed on tensors so it can later compute derivatives with respect to them.

```python
x = tf.Variable(3.0)

with tf.GradientTape() as tape:
    y = x ** 2

dy_dx = tape.gradient(y, x)   # dy/dx = 2x = 6.0
print(dy_dx.numpy())          # 6.0
```

This is exactly what powers Gradient Descent under the hood (see `neural_network.md` → Gradient Descent) — Keras's `.fit()` uses `GradientTape` internally so you don't have to write it yourself for standard training.

### 4. Variables vs Constants

| | `tf.constant` | `tf.Variable` |
|---|---|---|
| Mutability | Immutable — value cannot change | Mutable — value can be updated |
| Typical use | Fixed input data, hyperparameters | Model weights and biases (updated during training) |

---

## Building Models with Keras

Keras offers **three ways** to build a model, in increasing order of flexibility:

### 1. Sequential API (simplest)

Best for a simple stack of layers, each with exactly one input and one output.

```python
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    layers.Dense(16, activation='relu', input_shape=(10,)),
    layers.Dense(8, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])
```

### 2. Functional API (flexible)

Best when the model has **multiple inputs/outputs**, shared layers, or non-linear topology (branches, skip connections).

```python
from tensorflow.keras import Input, Model

inputs = Input(shape=(10,))
x = layers.Dense(16, activation='relu')(inputs)
x = layers.Dense(8, activation='relu')(x)
outputs = layers.Dense(1, activation='sigmoid')(x)

model = Model(inputs=inputs, outputs=outputs)
```

### 3. Model Subclassing (most flexible)

Best for research/custom architectures where you need full control over the forward pass — write a Python class like any other object-oriented model.

```python
class MyModel(keras.Model):
    def __init__(self):
        super().__init__()
        self.dense1 = layers.Dense(16, activation='relu')
        self.dense2 = layers.Dense(8, activation='relu')
        self.out = layers.Dense(1, activation='sigmoid')

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return self.out(x)

model = MyModel()
```

---

## Commonly Used Keras Layers

| Layer | Purpose |
|---|---|
| `Dense` | Fully connected layer — every neuron connects to every input |
| `Conv2D` | Convolutional layer for image data (see CNN in `neural_network.md`) |
| `MaxPooling2D` | Downsamples feature maps, reduces spatial dimensions |
| `LSTM` / `GRU` | Recurrent layers for sequential data |
| `Dropout` | Regularization — randomly deactivates neurons during training |
| `BatchNormalization` | Normalizes layer outputs for stable, faster training |
| `Flatten` | Converts a multi-dimensional tensor into a 1D vector (e.g., before a `Dense` layer) |
| `Embedding` | Converts integer word/token indices into dense vectors (used in NLP) |

---

## The Standard Keras Training Workflow

Every Keras model follows the same three-step pattern: **Compile → Fit → Evaluate/Predict**.

### Step 1: Compile

Configures the model for training by specifying the **optimizer**, **loss function**, and **metrics** to track.

```python
model.compile(
    optimizer='adam',              # see Gradient Descent optimizers in neural_network.md
    loss='binary_crossentropy',    # see Loss Functions in neural_network.md
    metrics=['accuracy']
)
```

### Step 2: Fit (Train)

Trains the model on data for a given number of epochs.

```python
history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=32,
    validation_split=0.2,     # holds out 20% of training data for validation
    verbose=1
)
```

### Step 3: Evaluate and Predict

```python
loss, accuracy = model.evaluate(X_test, y_test)
predictions = model.predict(X_new)
```

---

## Callbacks — Controlling Training Behavior

**Callbacks** are functions that run automatically at certain points during training (start/end of epoch, batch, etc.), commonly used to monitor or control the training process.

| Callback | Purpose |
|---|---|
| `EarlyStopping` | Stops training once a monitored metric (e.g., validation loss) stops improving — prevents overfitting |
| `ModelCheckpoint` | Automatically saves the model (or only the best version) during training |
| `ReduceLROnPlateau` | Reduces the learning rate when a metric plateaus |
| `TensorBoard` | Logs metrics for visualization in TensorBoard |

```python
from tensorflow.keras.callbacks import EarlyStopping

early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

model.fit(X_train, y_train, epochs=50, validation_split=0.2, callbacks=[early_stop])
```

---

## Saving and Loading Models

| Method | Description |
|---|---|
| `model.save('my_model.keras')` | Saves the entire model — architecture, weights, and optimizer state |
| `keras.models.load_model('my_model.keras')` | Reloads a previously saved model |
| `model.save_weights('weights.h5')` | Saves only the trained weights (architecture must be redefined separately) |

```python
model.save('my_model.keras')
loaded_model = keras.models.load_model('my_model.keras')
```

---

## Full Example: Image Classification with CNN (Keras)

```python
import tensorflow as tf
from tensorflow.keras import layers, models

# Load and preprocess data (e.g., MNIST handwritten digits)
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train, X_test = X_train / 255.0, X_test / 255.0          # normalize pixel values
X_train = X_train[..., None]                                # add channel dimension
X_test = X_test[..., None]

# Build a CNN
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')      # 10 classes (digits 0-9)
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(X_train, y_train, epochs=5, batch_size=64, validation_split=0.1)

test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc:.4f}")
```

---

## GPU / Hardware Acceleration

TensorFlow automatically uses a GPU if one is available and correctly configured (via CUDA/cuDNN for NVIDIA GPUs).

```python
print("GPUs available:", tf.config.list_physical_devices('GPU'))
```

- TensorFlow will place operations on the GPU automatically when possible.
- For multi-GPU or distributed training, TensorFlow provides `tf.distribute.Strategy` (e.g., `MirroredStrategy` for multiple GPUs on one machine).

---

## TensorFlow/Keras vs PyTorch — Quick Note

| | TensorFlow / Keras | PyTorch |
|---|---|---|
| **API style** | Keras = declarative, high-level | More "Pythonic", imperative by default |
| **Deployment** | Very mature production tooling (TF Serving, TFLite, TF.js) | Growing (TorchServe, ONNX export) |
| **Learning curve** | Keras is very beginner-friendly | Slightly more low-level, but flexible |
| **Popularity** | Widely used in industry/production | Widely used in research |

Both are capable of building the exact same models (Dense, CNN, RNN, Transformer, etc.) — the choice is largely about ecosystem and team preference.

---

## Frequently Asked Questions (FAQs)

**Q1. Is Keras a separate library from TensorFlow?**
Historically, Keras was a standalone library that could run on top of multiple backends (TensorFlow, Theano, CNTK). Since TensorFlow 2.0, Keras has been tightly integrated as `tensorflow.keras` and is the officially recommended way to use it — this is what "Keras" refers to in almost all modern code.

**Q2. What is the difference between `model.fit()` and a custom training loop?**
`model.fit()` is a high-level Keras method that handles the entire training loop (forward pass, loss computation, backpropagation via `GradientTape`, weight updates) automatically. A **custom training loop** — written manually using `tf.GradientTape` — gives full control over each step, useful for research scenarios `.fit()` doesn't support out of the box (e.g., custom loss computation logic, GANs with alternating updates).

**Q3. What does `input_shape` mean in the first layer?**
It tells the model the shape of a **single** input sample (excluding the batch dimension), so Keras can initialize the weight matrices correctly. For example, `input_shape=(10,)` means each sample has 10 features; Keras automatically handles batching on top of that.

**Q4. Why normalize input data (e.g., dividing pixel values by 255)?**
Neural networks train faster and more reliably when input features are on a similar, small scale (see Gradient Descent in `neural_network.md`) — large, unscaled inputs can cause unstable gradients and slower convergence.

**Q5. What is `sparse_categorical_crossentropy` vs `categorical_crossentropy`?**
Both are multi-class classification losses. Use `sparse_categorical_crossentropy` when labels are given as plain integers (e.g., `3`), and `categorical_crossentropy` when labels are **one-hot encoded** (e.g., `[0,0,0,1,0,...]`).
