# Neural Networks in Machine Learning

> Notes written in a GeeksforGeeks-style tutorial format — simple definitions first, then deeper detail, tables, formulas, and code examples.

---

## What is a Neural Network?

A **Neural Network** is a computational model inspired by the way biological neural networks in the human brain process information. It is a collection of connected units or nodes called **artificial neurons**, organized in layers, which loosely mimic the neurons in a biological brain.

Neural Networks are the foundation of **Deep Learning** and are used to recognize patterns, classify data, make predictions, and solve complex problems that are difficult to solve using traditional rule-based programming — such as image recognition, speech recognition, and natural language processing.

In simple words: a Neural Network takes an input, passes it through one or more layers of interconnected nodes (each of which performs a small computation), and produces an output. During training, the network **learns** by adjusting the strength of connections between nodes so that its output gets closer to the desired result.

---

## Biological Neuron vs Artificial Neuron

| Biological Neuron | Artificial Neuron |
|---|---|
| Dendrites receive signals | Inputs (`x1, x2, ..., xn`) receive data |
| Cell body (soma) processes signals | Summation function combines weighted inputs |
| Axon carries output signal | Activation function produces output |
| Synapse strength = signal importance | Weight (`w`) = importance of an input |
| Neuron fires if signal exceeds threshold | Activation function fires if weighted sum exceeds a threshold/bias |

The **Perceptron**, introduced by Frank Rosenblatt(father of deep learning) in 1958, was the first mathematical model of an artificial neuron and is the basic building block of a neural network.

---

## Basic Structure of a Neural Network

A typical neural network consists of **three types of layers**:

1. **Input Layer** — receives the raw input features (e.g., pixel values of an image, words of a sentence). It does not perform any computation; it simply passes data to the next layer.
2. **Hidden Layer(s)** — layers between the input and output layers where the actual computation happens. A network can have one hidden layer (**shallow network**) or many hidden layers (**deep network**, hence "Deep Learning").
3. **Output Layer** — produces the final result, such as a class label (classification) or a continuous value (regression).

```
Input Layer        Hidden Layer(s)        Output Layer
   (x1)  ─────┐        ┌──►(h1)──┐
   (x2)  ─────┼──►  ───┤         ├──►  ───►  (y)
   (x3)  ─────┘        └──►(h2)──┘
```

Each connection between two neurons has an associated **weight**, and each neuron (except input) has an associated **bias**. Together, weights and biases are the **parameters** the network learns during training.

---

## How a Single Neuron Works

Every artificial neuron performs two steps:

### Step 1: Weighted Sum (Linear Combination)

```
z = (w1*x1 + w2*x2 + ... + wn*xn) + b
```

Where:
- `x1, x2, ..., xn` → input values
- `w1, w2, ..., wn` → weights associated with each input
- `b` → bias term (allows shifting the activation threshold)
- `z` → weighted sum (also called the **pre-activation** value)

### Step 2: Activation Function

```
output = f(z)
```

The activation function `f` introduces **non-linearity** into the network. Without it, no matter how many layers are stacked, the network would behave like a single linear function — unable to learn complex patterns.

---

## Activation Functions

Activation functions decide whether a neuron should be "activated" or not, based on its weighted input.

| Activation Function | Formula | Range | Typical Use |
|---|---|---|---|
| **Sigmoid** | `f(z) = 1 / (1 + e^-z)` | (0, 1) | Binary classification output layer |
| **Tanh** | `f(z) = (e^z - e^-z) / (e^z + e^-z)` | (-1, 1) | Hidden layers (zero-centered, better than sigmoid) |
| **ReLU** (Rectified Linear Unit) | `f(z) = max(0, z)` | [0, ∞) | Most common choice for hidden layers |
| **Leaky ReLU** | `f(z) = z if z>0 else 0.01z` | (-∞, ∞) | Fixes "dying ReLU" problem |
| **Softmax** | `f(z_i) = e^(z_i) / Σ e^(z_j)` | (0, 1), sums to 1 | Multi-class classification output layer |

**Note:** ReLU is the most widely used activation function in hidden layers because it is computationally cheap and helps mitigate the **vanishing gradient problem**, which was common with Sigmoid and Tanh in deep networks.

---

## Forward Propagation

**Forward Propagation** is the process of passing input data through the network, layer by layer, to generate an output/prediction.

1. Input values enter the input layer.
2. Each hidden layer computes a weighted sum of its inputs, adds a bias, and applies an activation function.
3. This process repeats until the output layer produces the final prediction.

```
Layer 1: z1 = W1·X + b1        a1 = f(z1)
Layer 2: z2 = W2·a1 + b2       a2 = f(z2)
Output:  z3 = W3·a2 + b3       y_pred = f(z3)
```

---

## Loss Function (Cost Function)

The **Loss Function** measures how far the network's prediction (`y_pred`) is from the actual target value (`y_actual`). The goal of training is to **minimize** this loss.

| Loss Function | Formula | Used For |
|---|---|---|
| **Mean Squared Error (MSE)** | `(1/n) Σ (y_actual - y_pred)^2` | Regression problems |
| **Binary Cross-Entropy** | `-[y·log(p) + (1-y)·log(1-p)]` | Binary classification |
| **Categorical Cross-Entropy** | `-Σ y_i · log(p_i)` | Multi-class classification |

---

## Gradient Descent

**Gradient Descent** is an **optimization algorithm** used to minimize the loss function by iteratively adjusting the network's weights and biases in the direction that reduces the error the most.

### Intuition

Imagine you are standing on a hilly surface (the **loss landscape**) in thick fog, and your goal is to reach the lowest point (**minimum loss**) without being able to see the whole terrain. The only thing you can sense is the **slope** under your feet. A sensible strategy is to take a small step in the direction that goes downhill the fastest, then repeat. That is exactly what Gradient Descent does — the "slope" is the **gradient** (derivative) of the loss function, and the network keeps stepping downhill until it reaches (close to) the minimum.

### The Formula

For every weight `w` in the network:

```
w_new = w_old - learning_rate * (∂Loss / ∂w)
```

Where:
- `∂Loss / ∂w` → the **gradient** — how much the loss changes when `w` changes. It tells us the **direction of steepest ascent**, so we move in the **opposite** direction (hence the minus sign) to decrease the loss.
- `learning_rate` (α) → a small positive number that controls **how big a step** to take in that direction.

This same update rule is applied to every weight and bias in the network, using the gradients computed during **Backpropagation** (see below).

### Role of the Learning Rate

The learning rate is one of the most important hyperparameters to get right:

| Learning Rate | Effect |
|---|---|
| **Too small** | Training converges very slowly, may get stuck before reaching a good minimum, wastes compute |
| **Too large** | Updates overshoot the minimum, loss can bounce around or even diverge (increase instead of decrease) |
| **Just right** | Loss decreases smoothly and steadily until it converges to a minimum |

In practice, techniques like **learning rate schedules** (gradually decreasing the learning rate over training) or **adaptive optimizers** (Adam, RMSprop) are used instead of a single fixed value.

### Types of Gradient Descent

| Variant | How much data used per update | Description |
|---|---|---|
| **Batch Gradient Descent** | Entire training dataset | Computes the gradient using **all** training examples before making one weight update — accurate and stable, but slow and memory-heavy on large datasets |
| **Stochastic Gradient Descent (SGD)** | One example | Updates weights after **every single** training example — very fast per step, but the path to the minimum is noisy/zig-zagging |
| **Mini-Batch Gradient Descent** | A small batch (e.g., 32, 64, 128) | Updates weights after each small batch of examples — the most widely used approach in practice, balancing the stability of Batch GD with the speed of SGD |

### Common Challenges in Gradient Descent

- **Local Minima** — the algorithm can get stuck in a valley that isn't the lowest point overall (the **global minimum**).
- **Saddle Points** — flat regions where the gradient is close to zero in some directions but not a true minimum, which can slow down learning significantly.
- **Vanishing/Exploding Gradients** — in deep networks, gradients can become extremely small (vanish) or extremely large (explode) as they are backpropagated through many layers, making updates ineffective or unstable.
- **Choosing the Learning Rate** — as covered above, this single hyperparameter can make or break training.

### Optimizers Built on Gradient Descent

Modern training rarely uses plain Gradient Descent as-is — it uses improved variants ("**optimizers**") that adapt the step size or direction using information from past gradients:

- **SGD with Momentum** — accelerates convergence by adding a fraction of the previous update to the current one, helping push through small local bumps and flat regions
- **Adagrad** — adapts the learning rate **per parameter**, giving smaller updates to frequently-updated parameters
- **RMSprop** — fixes Adagrad's aggressively/monotonically decaying learning rate using a moving average of squared gradients
- **Adam** (Adaptive Moment Estimation) — combines Momentum + RMSprop; the most widely used optimizer in modern deep learning due to its fast, stable convergence with little tuning

### Simple Gradient Descent Example in Python

```python
# Minimizing a simple function f(x) = x^2 using Gradient Descent
x = 10.0            # initial guess
learning_rate = 0.1
epochs = 20

for i in range(epochs):
    gradient = 2 * x          # derivative of x^2 is 2x
    x = x - learning_rate * gradient
    print(f"Epoch {i+1}: x = {x:.4f}")

# x keeps moving closer to 0, which is the minimum of f(x) = x^2
```

---

## Backpropagation

**Backpropagation** (short for "backward propagation of errors") is the algorithm used to train neural networks. It calculates how much each weight contributed to the error and updates the weights to reduce that error.

**How it works, step by step:**

1. Perform **forward propagation** to get the predicted output.
2. Calculate the **loss** using a loss function.
3. Compute the **gradient** (derivative) of the loss with respect to each weight, moving backward from the output layer to the input layer, using the **chain rule** of calculus.
4. Update each weight using **Gradient Descent** (see above):

```
w_new = w_old - learning_rate * (∂Loss / ∂w)
```

This process is repeated over many iterations (**epochs**) until the loss becomes acceptably small. In short: **Backpropagation** computes the gradients, and **Gradient Descent** uses those gradients to actually update the weights.

---

## Types of Neural Networks

Neural networks come in many architectures, each suited to a different kind of data or problem — the same underlying building blocks (neurons, weights, activation functions, backpropagation) arranged with a different **inductive bias**.

| Type | Best Suited For | Key Feature |
|---|---|---|
| FNN / MLP | Tabular data, simple classification | One-directional flow of data |
| CNN | Images, video | Convolution + pooling for spatial features |
| RNN | Sequential/time-series data | Feedback loop retains memory |
| LSTM/GRU | Long sequences | Gating mechanism solves vanishing gradients |
| Autoencoder | Unsupervised feature learning | Encoder-decoder reconstruction |
| GAN | Data/image generation | Generator vs Discriminator competition |
| Transformer | NLP, LLMs | Self-attention, parallel processing |
| RBFN | Function approximation | Distance-based radial basis activation |
| Modular NN | Large/complex tasks split into sub-problems | Independent specialized sub-networks |
| DBN / Boltzmann Machine | Unsupervised pre-training, feature learning | Stacked RBMs, layer-wise greedy training |
| GNN | Graph-structured data | Message passing between connected nodes |
| Siamese Network | Similarity/verification tasks | Shared-weight twin networks + distance comparison |

> This is just a quick-reference summary. For the full breakdown of each architecture — how it works, diagrams, advantages/limitations, and applications — see the dedicated notes in `types_of_neural_networks.md`. For a deep dive specifically on Transformers/LLMs (tokenization, self-attention formula, encoder vs decoder-only, KV cache, MoE, training pipeline), see `llm_architecture.md`.

---

## Overfitting and Regularization Techniques

**Overfitting** occurs when a neural network learns the training data too well — including its noise — and performs poorly on unseen/test data.

| Technique | How it Helps |
|---|---|
| **Dropout** | Randomly "drops" (deactivates) a fraction of neurons during training, forcing the network to not rely too heavily on specific neurons |
| **L1/L2 Regularization** | Adds a penalty term to the loss function based on the magnitude of weights, discouraging overly large weights |
| **Batch Normalization** | Normalizes the output of a layer before passing it to the next, stabilizing and speeding up training |
| **Early Stopping** | Stops training once validation loss stops improving, before the model starts overfitting |
| **Data Augmentation** | Artificially increases training data variety (e.g., rotating/flipping images) so the model generalizes better |

---

## Hyperparameters in a Neural Network

Hyperparameters are configuration values set **before** training begins (as opposed to weights/biases, which are **learned**).

| Hyperparameter | Description |
|---|---|
| **Learning Rate** | Controls the step size of weight updates during gradient descent |
| **Number of Epochs** | Number of times the entire training dataset passes through the network |
| **Batch Size** | Number of training samples processed before weights are updated |
| **Number of Hidden Layers/Neurons** | Determines the depth and capacity of the network |
| **Activation Function** | Determines how neurons introduce non-linearity |
| **Optimizer** | Algorithm used to update weights (SGD, Adam, RMSprop, etc.) |

---

## Advantages of Neural Networks

- Can model highly **complex, non-linear relationships** in data.
- **Automatically learn features** from raw data (especially CNNs on images), reducing the need for manual feature engineering.
- Highly **flexible** — applicable across domains: vision, text, audio, tabular data.
- **Scale well** with large amounts of data — performance often keeps improving as more data is added.

## Disadvantages of Neural Networks

- Require **large amounts of labeled data** to perform well.
- **Computationally expensive** to train — often need GPUs/TPUs.
- Act as a **"black box"** — difficult to interpret why a specific prediction was made.
- Prone to **overfitting** if not regularized properly.
- Sensitive to **hyperparameter choices** (learning rate, architecture, etc.).

---

## Applications of Neural Networks

| Domain | Example Use Case |
|---|---|
| Computer Vision | Face recognition, object detection, medical imaging |
| Natural Language Processing | Chatbots, machine translation, sentiment analysis |
| Speech | Voice assistants, speech-to-text |
| Finance | Fraud detection, stock price prediction, credit scoring |
| Healthcare | Disease diagnosis, drug discovery |
| Autonomous Systems | Self-driving cars, robotics |
| Recommendation Systems | Product/content recommendations (Netflix, Amazon, YouTube) |

---

## Simple Implementation in Python (Keras)

Below is a simple feedforward neural network built using **TensorFlow/Keras** for a binary classification problem.

```python
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

# Sample data: X = features, y = binary labels
X = np.random.rand(1000, 10)
y = np.random.randint(0, 2, size=(1000, 1))

# Build the model
model = keras.Sequential([
    layers.Dense(16, activation='relu', input_shape=(10,)),  # hidden layer 1
    layers.Dense(8, activation='relu'),                       # hidden layer 2
    layers.Dense(1, activation='sigmoid')                      # output layer
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train the model
model.fit(X, y, epochs=10, batch_size=32, validation_split=0.2)

# Predict
predictions = model.predict(X[:5])
print(predictions)
```

**What this code does:**
1. Creates a `Sequential` model — a linear stack of layers.
2. Adds two hidden layers with **ReLU** activation.
3. Adds an output layer with **Sigmoid** activation (since this is binary classification).
4. Compiles the model with the **Adam** optimizer and **binary cross-entropy** loss.
5. Trains the model using `.fit()`, which internally performs forward propagation, loss calculation, and backpropagation for the given number of epochs.

---

## Neural Network vs Deep Learning

| Neural Network | Deep Learning |
|---|---|
| General term for any network of connected neurons | A **subset** of neural networks with many hidden layers (deep architecture) |
| Can be shallow (1 hidden layer) | Always has multiple (often dozens/hundreds of) hidden layers |
| Good for simpler problems | Needed for complex tasks like image recognition, NLP, LLMs |

---

## Frequently Asked Questions (FAQs)

**Q1. What is the difference between a Perceptron and a Neural Network?**
A Perceptron is a single-layer, single-neuron model that can only solve linearly separable problems. A Neural Network is a network of many perceptron-like neurons arranged in multiple layers, capable of solving complex, non-linear problems.

**Q2. Why do we need activation functions?**
Activation functions introduce non-linearity into the network. Without them, stacking multiple layers would be mathematically equivalent to a single linear layer, severely limiting what the network could learn.

**Q3. What is the vanishing gradient problem?**
It occurs when gradients become extremely small as they are backpropagated through many layers, causing earlier layers to learn very slowly or stop learning altogether. It is common with Sigmoid/Tanh activations in deep networks and is mitigated using ReLU, LSTM gating, batch normalization, or residual connections.

**Q4. How many hidden layers/neurons should a network have?**
There is no fixed rule — it depends on the complexity of the problem and dataset size. It's common to start with a small architecture and increase depth/width while monitoring validation performance, using techniques like cross-validation or hyperparameter tuning.

**Q5. What is the difference between an epoch, a batch, and an iteration?**
- **Epoch** — one complete pass through the entire training dataset.
- **Batch** — a subset of the training data processed together before a weight update.
- **Iteration** — one weight update step (i.e., one batch processed).

Suppose you have 1,000 training data points
epoch = model sees all 1,000 training examples once.
```
Dataset: 1,000 examples

Epoch 1 → sees all 1,000
Epoch 2 → sees all 1,000
Epoch 3 → sees all 1,000
...
Epoch 10 → sees all 1,000
```
batch = Instead of giving all 1,000 examples to the model at once, we divide them into smaller groups called batches
Total data = 1,000
Batch size = 100
```
Batch 1 → examples 1–100
Batch 2 → examples 101–200
Batch 3 → examples 201–300
...
Batch 10 → examples 901–1000
````
So one epoch contains 10 batches
iteration = one batch being processed by the model.
```1 iteration = process 1 batch = 100 examples```

```
1 Epoch
    ↓
Batch 1 → Iteration 1
Batch 2 → Iteration 2
Batch 3 → Iteration 3
...
Batch 10 → Iteration 10

2 Epoch
    ↓
Batch 1 → Iteration 11
Batch 2 → Iteration 12
...
Batch 10 → Iteration 20

3 Epoch
    ↓
Batch 1 → Iteration 21
...
Batch 10 → Iteration 30
```