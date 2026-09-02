# Types of Neural Networks

> Notes written in a GeeksforGeeks-style tutorial format — simple definitions first, then deeper detail, tables, and examples. Builds on the fundamentals in `neural_network.md` (neurons, layers, activation functions, forward/backward propagation). For a deep dive specifically on Transformers/LLMs, see `llm_architecture.md`.

---

## Why Are There So Many Types of Neural Networks?

There is no single "best" neural network architecture — the right choice depends entirely on the **shape of the data** and the **nature of the problem**:

- Tabular/structured data with no special ordering → a simple **Feedforward Network** is enough.
- Data with spatial structure (images) → **CNNs** exploit that structure.
- Data with sequential/temporal structure (text, time series) → **RNNs/LSTMs/Transformers** exploit order and context.
- No labeled data at all → **Autoencoders/GANs/DBNs** learn from the data's own structure.
- Data with relational structure (graphs) → **GNNs** operate directly on nodes and edges.

Each architecture is really the same underlying building blocks (neurons, weights, activation functions, backpropagation — see `neural_network.md`) arranged with a different **inductive bias**: an assumption baked into the architecture about what patterns are likely to matter, which makes learning faster and more accurate for that specific type of data.

---

## 1. Feedforward Neural Network (FNN / MLP)

A **Feedforward Neural Network**, also called a **Multi-Layer Perceptron (MLP)**, is the most basic type of artificial neural network. Information moves in **only one direction** — from the input layer, through one or more hidden layers, to the output layer — with **no loops, no cycles, and no memory** of previous inputs. Every prediction is made purely from the current input, independent of anything seen before.

```
Input Layer     Hidden Layer(s)     Output Layer
  (x1) ──┐         ┌──►(h1)──┐
  (x2) ──┼──► ──────┤         ├──► ──────► (y)
  (x3) ──┘         └──►(h2)──┘

          (data flows strictly left → right, never backward)
```

### How It Works

- Every neuron in a layer is connected to **every** neuron in the next layer — these are called **fully connected (Dense) layers**.
- Each connection carries a **weight**, and each neuron has a **bias**. A neuron computes a weighted sum of its inputs, adds the bias, then passes the result through an **activation function** (ReLU, Sigmoid, Tanh — see `neural_network.md`) to introduce non-linearity.
- The network is trained end-to-end using **Forward Propagation** (compute the prediction), a **Loss Function** (measure the error), and **Backpropagation + Gradient Descent** (update every weight to reduce that error) — all detailed in `neural_network.md`.

### Why the Hidden Layer Matters — the XOR Problem

A **single-layer Perceptron** (no hidden layer) can only learn patterns that are **linearly separable** — i.e., a straight line/plane can divide the classes. It famously **cannot** learn the XOR function (output is 1 only when inputs differ), because no single straight line can separate XOR's outputs.

| x1 | x2 | XOR |
|---|---|---|
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

Adding just **one hidden layer** lets the MLP combine multiple linear boundaries into a non-linear decision boundary, solving XOR easily. This is the whole reason "deep"/multi-layer networks exist rather than stopping at a single perceptron.

### Universal Approximation Theorem

A key theoretical result: an MLP with **just one hidden layer** (with enough neurons and a suitable non-linear activation) can, in principle, approximate **any continuous function** to arbitrary accuracy. In practice, though, going **deeper** (more layers, each smaller) is usually far more parameter-efficient and easier to train than going extremely **wide** with one huge hidden layer.

### Advantages

- Simple to understand, implement, and debug — the foundational architecture every other network builds on.
- Works well out-of-the-box on **structured/tabular data** where feature order/spatial position doesn't matter.
- Theoretically capable of approximating any function given enough capacity.

### Limitations

- Has **no memory** of previous inputs — unsuitable for sequences (text, time series) without external tricks.
- **No spatial awareness** — if used directly on an image, it must flatten the 2D grid into a 1D vector, throwing away information about which pixels are near each other.
- Fully connected layers mean a **huge number of parameters** for high-dimensional inputs (e.g., a 224×224 image flattened has 150,528 input values), making it prone to overfitting and inefficient compared to CNNs for such data.

### Applications

Credit scoring, medical diagnosis from tabular lab results, basic regression/classification problems, simple recommendation scoring — any task where inputs are a fixed-size vector of independent features.

### Minimal Keras Example

```python
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    layers.Dense(16, activation='relu', input_shape=(10,)),
    layers.Dense(8, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])
```

---

## 2. Convolutional Neural Network (CNN)

A **Convolutional Neural Network** is a specialized architecture designed primarily for **image and other grid-like data** (video frames, spectrograms). Instead of connecting every neuron to every pixel — which would require an enormous, impractical number of parameters and would ignore spatial structure entirely — a CNN slides small learnable **filters (kernels)** across the input, automatically learning to detect local visual patterns like edges, textures, and shapes.

### The Convolution Operation

A **filter/kernel** (e.g., a small 3×3 grid of learnable weights) slides across the input image. At each position, it computes the **element-wise product** between the filter and the patch of the image it currently covers, then sums the result into a single number. Repeating this across the whole image produces a **feature map** — a 2D grid showing where in the image that particular pattern (e.g., a vertical edge) was detected.

```
Input (5x5)          Filter (3x3)         Feature Map (3x3)
1 1 1 0 0                                  4 3 4
0 1 1 1 0     ⨂       1 0 1                2 4 3
0 0 1 1 1              0 1 0                2 3 4
0 0 1 1 0              1 0 1
0 1 1 0 0
```

Three parameters control how convolution behaves:

| Parameter | Meaning |
|---|---|
| **Filter/Kernel size** | The spatial size of the sliding window (e.g., 3×3, 5×5) |
| **Stride** | How many pixels the filter moves at each step (stride 1 = move one pixel at a time; larger stride = more downsampling) |
| **Padding** | Whether to pad the input's border with zeros — **"same"** padding keeps output size equal to input size; **"valid"** padding uses no padding, so the output shrinks |

Output feature-map size formula: `Output = ((W − F + 2P) / S) + 1`, where `W` = input width, `F` = filter size, `P` = padding, `S` = stride.

### Key Layers

| Layer | Role |
|---|---|
| **Convolution Layer** | Slides learnable filters over the input to produce **feature maps**, detecting local patterns |
| **Activation (ReLU)** | Applied after convolution to introduce non-linearity, same role as in any neural network (see `neural_network.md`) |
| **Pooling Layer** (Max/Average) | Downsamples feature maps — **Max Pooling** keeps only the strongest activation in each small region — reducing size and computation while retaining the most important signal |
| **Fully Connected Layer** | Final `Dense` layers that combine all extracted features to make the actual classification/regression prediction |

### Hierarchical Feature Learning

Stacking multiple convolution+pooling blocks lets a CNN build up a **hierarchy of features**: early layers detect simple patterns (edges, colors, corners), middle layers combine these into parts (eyes, wheels, textures), and deeper layers combine parts into whole objects (faces, cars). This hierarchical composition is a big part of why CNNs are so effective for vision tasks, and it happens automatically through training — no manual feature engineering required.

```
Raw Pixels → Edges → Textures/Shapes → Object Parts → Full Objects → Class Prediction
   (Layer 1)  (Layer 2)     (Layer 3)      (Layer 4)      (Output)
```

### Why It Works So Well for Images

- **Local connectivity** — each neuron only looks at a small local patch of the input (its "receptive field"), matching the intuition that nearby pixels are more related than distant ones.
- **Parameter sharing** — the *same* filter (same weights) is reused across every position in the image, so a "vertical edge detector" learned in one corner works everywhere else too. This massively reduces the number of trainable parameters compared to a fully connected network processing the same image.
- **Translation invariance** — because filters are applied everywhere, a CNN can recognize a feature (e.g., a cat's ear) regardless of where it appears in the image.

### Well-Known CNN Architectures

| Architecture | Notable For |
|---|---|
| **LeNet-5** | One of the earliest CNNs (1998), digit recognition |
| **AlexNet** | 2012 — sparked the deep learning boom by winning ImageNet by a huge margin |
| **VGG** | Very deep, simple uniform 3×3 filters stacked repeatedly |
| **ResNet** | Introduced **residual/skip connections**, enabling networks with 100+ layers to train successfully |
| **Inception/GoogLeNet** | Uses multiple filter sizes in parallel within the same layer for multi-scale feature extraction |

### Advantages

- Dramatically fewer parameters than an equivalent fully connected network on image data.
- Automatically learns relevant visual features — no manual feature engineering.
- Translation-invariant — robust to where an object appears in the frame.

### Limitations

- Still needs a large amount of labeled training data to perform well.
- Computationally intensive to train (benefits heavily from GPU/TPU acceleration).
- Not naturally invariant to rotation or scale changes (data augmentation is commonly used to compensate).

### Applications

Image classification, object detection, facial recognition, medical image analysis (X-rays, MRI/CT scans), self-driving car perception, video analysis.

### Minimal Keras Example

```python
from tensorflow.keras import layers, models

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])
```

---

## 3. Recurrent Neural Network (RNN)

A **Recurrent Neural Network** is designed for **sequential data** — text, time series, audio, speech — where the meaning of the current input depends on what came before it. Unlike an FNN, which treats every input as independent, an RNN maintains a **hidden state** that gets passed forward from one time step to the next, acting as a form of "memory" of everything the network has seen so far in the sequence.

### How It Works — Unrolled Through Time

At every time step `t`, the same RNN cell (same shared weights) takes two inputs: the current input `x_t` and the previous hidden state `h_{t-1}`, and produces a new hidden state `h_t`:

```
h_t = f(W_hh · h_{t-1}  +  W_xh · x_t  +  b_h)
y_t = g(W_hy · h_t + b_y)
```

Where `f` is typically `tanh`, and `W_hh`, `W_xh`, `W_hy` are weight matrices **shared across every time step** (this weight-sharing is what makes an RNN work on sequences of any length with a fixed number of parameters).

```
x1 → [RNN cell] → h1 ─┐
                       ├──► x2 → [RNN cell] → h2 ─┐
                                                    ├──► x3 → [RNN cell] → h3 → output
```

"Unrolling" the recurrent loop across time steps like this turns the RNN into what is effectively a very deep feedforward network — one layer per time step — which is key to understanding both how it's trained and why it struggles on long sequences.

### Training: Backpropagation Through Time (BPTT)

RNNs are trained using **Backpropagation Through Time** — the unrolled network (one "layer" per time step) is backpropagated through exactly like a deep FNN, and gradients for the shared weights are accumulated across all time steps before a single update.

### The Core Limitation: Vanishing/Exploding Gradients

Because BPTT effectively backpropagates through as many "layers" as there are time steps, gradients for long sequences must be multiplied repeatedly (via the chain rule — see `neural_network.md`). If these repeated multiplications are consistently **less than 1**, the gradient shrinks toward zero (**vanishing gradient**) — the network effectively "forgets" long-range dependencies and can't learn relationships between distant elements in a sequence. If they're consistently **greater than 1**, the gradient can grow uncontrollably (**exploding gradient**), causing unstable training. This is the single biggest limitation of plain RNNs, and it's the exact problem **LSTM/GRU** (see below) and **Transformers** were designed to fix.

### RNN Input/Output Patterns

RNNs are flexible in how inputs and outputs are structured across time:

| Pattern | Description | Example |
|---|---|---|
| **One-to-Many** | Single input → sequence output | Image captioning (one image → a sentence) |
| **Many-to-One** | Sequence input → single output | Sentiment analysis (a review → positive/negative) |
| **Many-to-Many (aligned)** | Sequence input → sequence output, same length | Part-of-speech tagging (word-by-word) |
| **Many-to-Many (unaligned)** | Sequence input → sequence output, different length | Machine translation (encoder-decoder RNN) |

### Bidirectional RNNs

A **Bidirectional RNN** runs two RNNs over the input — one reading left-to-right, one reading right-to-left — and combines both hidden states at each step. This lets the network use context from **both directions** (past and future) when processing a given position, useful whenever the full sequence is available upfront (e.g., analyzing a complete sentence, as opposed to real-time streaming generation).

### Advantages

- Naturally handles **variable-length sequences** using a fixed set of parameters (weight sharing across time steps).
- Captures order/temporal dependencies that a plain FNN completely ignores.

### Limitations

- **Vanishing/exploding gradients** make it hard to learn long-range dependencies (see above).
- **Sequential by nature** — each step depends on the previous one, so RNNs cannot be parallelized across the time dimension the way CNNs/Transformers can, making them slow to train on long sequences.
- Effectively has only **short-term memory** in practice, which is exactly why LSTM/GRU (and eventually Transformers) were developed.

### Applications

Basic sequence modeling, early speech recognition systems, simple time-series forecasting. (For anything beyond short sequences, LSTM/GRU or Transformer architectures are used today instead — see below and `llm_architecture.md`.)

### Minimal Keras Example

```python
from tensorflow.keras import layers, models

model = models.Sequential([
    layers.SimpleRNN(32, input_shape=(timesteps, features)),
    layers.Dense(1, activation='sigmoid')
])
```

---

## 4. Long Short-Term Memory (LSTM) / GRU

Special types of RNNs specifically designed to overcome the vanishing gradient problem, using **gates** that learn what information to keep, forget, or output at each step.

| Gate (LSTM) | Purpose |
|---|---|
| **Forget Gate** | Decides what information to discard from the cell state |
| **Input Gate** | Decides what new information to add to the cell state |
| **Output Gate** | Decides what part of the cell state to output as the hidden state |

**GRU (Gated Recurrent Unit)** is a simplified variant that merges the forget and input gates into a single **update gate**, making it computationally cheaper than LSTM while often achieving comparable performance.

- **Applications:** Language modeling, machine translation, speech recognition, stock price prediction — anywhere long-range sequence dependencies matter but a full Transformer isn't needed/available.

---

## 5. Autoencoders

An **unsupervised** neural network trained to reconstruct its own input, forcing it to learn a compressed, meaningful representation of the data along the way.

```
Input → [Encoder] → Latent/Bottleneck Representation → [Decoder] → Reconstructed Output
```

- **Encoder:** Compresses the input into a smaller **latent representation** (the "bottleneck").
- **Decoder:** Reconstructs the original input from that compressed representation.
- **Why the bottleneck matters:** By forcing the network to squeeze information through a smaller layer and still reconstruct the original input, it's forced to learn only the **most important, non-redundant features** of the data.
- **Applications:** Dimensionality reduction, anomaly detection (poor reconstruction = anomaly), denoising (train on noisy → clean pairs), image compression.

---

## 6. Generative Adversarial Network (GAN)

A **Generative Adversarial Network** consists of **two neural networks trained together in direct competition**, each trying to outdo the other — a setup inspired by a "forger vs detective" analogy:

- **Generator** — takes random noise as input and tries to generate fake data that looks indistinguishable from real data.
- **Discriminator** — a binary classifier that tries to tell apart the Generator's fake samples from real training samples.

```
Random Noise (z) → [Generator (G)] → Fake Sample ──┐
                                                      ├──► [Discriminator (D)] → Real or Fake? (probability)
                   Real Sample ────────────────────── ┘
```

### The Adversarial (Minimax) Training Process

GANs are trained using a **minimax game** between the two networks, formalized as:

```
min_G  max_D   E[log D(x)]  +  E[log(1 − D(G(z)))]
```

In plain terms: the **Discriminator (D)** is trained to **maximize** its ability to correctly label real data as real and fake data as fake, while the **Generator (G)** is trained to **minimize** the Discriminator's success — i.e., to fool it as often as possible. Training alternates between the two:

1. **Train the Discriminator** for a step — show it a batch of real samples and a batch of the Generator's current fake samples, and update its weights (via backpropagation, like any network) to get better at telling them apart.
2. **Train the Generator** for a step — generate fake samples and update the Generator's weights to *increase* the Discriminator's error rate (i.e., make fakes that fool it), while the Discriminator's weights stay fixed during this step.
3. **Repeat**, with both networks gradually improving in tandem — ideally converging to a point where the Generator's outputs are so realistic the Discriminator can do no better than random guessing (50/50).

### Common Training Challenges

| Challenge | Description |
|---|---|
| **Mode Collapse** | The Generator finds a small set of outputs that reliably fool the Discriminator and keeps producing only those, losing diversity instead of covering the full variety of real data |
| **Training Instability** | Because it's a two-player game rather than a single loss being minimized, GAN training can oscillate or fail to converge if the Generator and Discriminator become mismatched in strength |
| **Vanishing Gradients** | If the Discriminator becomes too good too quickly, it gives the Generator almost no useful gradient signal to improve from |

### Notable GAN Variants

| Variant | Key Idea |
|---|---|
| **DCGAN** | Uses convolutional layers (instead of plain dense layers) for image generation — the standard baseline GAN architecture for images |
| **Conditional GAN (cGAN)** | Both G and D are also given a class label/condition, allowing controlled generation (e.g., "generate a fake image of digit 7") |
| **CycleGAN** | Learns to translate images between two domains (e.g., horse ↔ zebra) **without** needing paired training examples |
| **StyleGAN** | Introduces style-based control over generated image features at different levels of detail, producing highly realistic, high-resolution faces |
| **WGAN (Wasserstein GAN)** | Replaces the original loss with the Wasserstein distance, significantly improving training stability |

### Advantages

- Capable of generating highly realistic, novel data (images, audio, text) — genuinely creative output, not just classification.
- **Unsupervised** — doesn't require labeled data, since it learns purely from the distribution of real examples.
- Broadly applicable across data types and highly extensible (see variants above).

### Limitations

- Notoriously **unstable and tricky to train** — very sensitive to hyperparameters and the relative "strength" of G vs D.
- **Mode collapse** can silently reduce output diversity.
- Hard to **evaluate objectively** — there's no simple loss number that directly measures "how realistic/diverse are the outputs," often requiring human judgment or specialized metrics (e.g., FID score).

### Applications

Photorealistic image generation, deepfakes, data augmentation (synthetic training data for other models), image super-resolution, image-to-image translation (style transfer, sketch-to-photo), art and design generation.

### Minimal Keras Skeleton

```python
from tensorflow.keras import layers, models

# Generator: noise -> fake image
generator = models.Sequential([
    layers.Dense(128, activation='relu', input_shape=(100,)),  # 100-dim noise vector
    layers.Dense(784, activation='tanh')                        # e.g., flattened 28x28 image
])

# Discriminator: image -> real/fake probability
discriminator = models.Sequential([
    layers.Dense(128, activation='relu', input_shape=(784,)),
    layers.Dense(1, activation='sigmoid')
])
```

---

## 7. Transformer Networks

A **Transformer** is an architecture that relies entirely on a **self-attention mechanism** instead of recurrence (RNN) or convolution (CNN), allowing the model to process an **entire sequence in parallel** and let every element directly relate to every other element, regardless of how far apart they are in the sequence. Introduced in the 2017 paper *"Attention Is All You Need,"* it is now the dominant architecture behind essentially all modern large language models.

### Why Transformers Replaced RNNs for Sequence Tasks

RNNs process a sequence **one step at a time**, passing a hidden state forward — which creates two hard problems: it's inherently **sequential** (can't be parallelized across time, so training is slow), and it suffers from **vanishing gradients** over long sequences (see RNN section above), making distant relationships hard to learn. The Transformer fixes both: every token attends to every other token in a **single step**, so there's no long chain of sequential updates to vanish through, and because there's no step-by-step dependency, the whole sequence can be processed **simultaneously** on a GPU.

```
Input Tokens → Embeddings + Positional Encoding
             → [ Self-Attention → Feed-Forward ] × N layers
             → Output
```

### The Self-Attention Mechanism (Core Idea)

For every token, the model computes three vectors — a **Query (Q)**, a **Key (K)**, and a **Value (V)** — by multiplying its embedding with learned weight matrices. Attention scores are computed by comparing each token's Query against every token's Key, converted into weights via **softmax**, and used to take a weighted combination of all the Value vectors:

```
Attention(Q, K, V) = softmax( (Q · K^T) / √d_k ) · V
```

Intuitively: for each word, the model asks "which other words in this sequence are relevant to understanding me?" and blends in information from those words proportionally to how relevant they are — e.g., in *"The animal didn't cross the street because **it** was tired,"* attention lets "it" directly connect to "animal" no matter how far apart they sit in the sentence.

### Multi-Head Attention

Rather than computing attention once, the Transformer runs **several attention "heads" in parallel**, each with its own learned Q/K/V weights — letting different heads specialize in different types of relationships (e.g., grammatical structure vs. long-range topical references) simultaneously. Their outputs are concatenated and combined into the final representation.

### Positional Encoding

Because self-attention looks at all tokens at once with no inherent sense of order, a **positional encoding** vector (representing each token's position in the sequence) is added to its embedding before attention is applied — otherwise, the model couldn't distinguish "dog bites man" from "man bites dog".

### The Full Transformer Block

Each Transformer layer wraps self-attention and a small feed-forward network with **residual (skip) connections** and **layer normalization**, then stacks many such blocks (e.g., dozens to over a hundred in large LLMs) to build up deep, richly contextual representations:

```
        ┌─────────────────────────────┐
        │   Add & LayerNorm            │
        │        ↑                     │
        │   Feed-Forward Network       │
        │        ↑                     │
        │   Add & LayerNorm            │
        │        ↑                     │
        │   Multi-Head Self-Attention  │
        │        ↑                     │
Input ──┴─────────────────────────────┘── Output
              (repeated N times)
```

### Encoder-only vs Decoder-only vs Encoder-Decoder

| Type | Attention | Example Models | Best For |
|---|---|---|---|
| **Encoder-only** | Bidirectional (sees full context both ways) | BERT | Understanding tasks — classification, embeddings, search |
| **Decoder-only** | Causal/masked (only sees previous tokens) | GPT, Claude, Llama | Text generation — chat, completion (most modern LLMs) |
| **Encoder-Decoder** | Encoder bidirectional, Decoder causal + attends to encoder | T5, original Transformer | Sequence-to-sequence — translation, summarization |

### Advantages

- **Fully parallelizable** — trains far faster than RNNs on modern GPU/TPU hardware since there's no sequential bottleneck.
- **Superior long-range dependency handling** — any two tokens connect in a single attention step, unlike RNNs where distant tokens must pass through many sequential hidden-state updates.
- Scales extremely well with more data and parameters (see scaling laws in `llm_architecture.md`), which is a major reason today's largest, most capable models are all Transformer-based.

### Limitations

- Self-attention has **O(n²)** time/memory complexity in sequence length — doubling the input length roughly quadruples the attention computation, making very long contexts expensive.
- Requires large amounts of data and compute to train well from scratch.
- No inherent notion of order (must be explicitly added via positional encoding).

### Applications

Large Language Models (GPT, BERT, Claude), machine translation, text summarization, code generation, and — via Vision Transformers (ViT) — increasingly image tasks as well.

> This is a solid grounding in the architecture itself. For the full LLM-specific breakdown — tokenization, embeddings, the complete self-attention walkthrough with code, KV cache, Mixture of Experts, and how LLMs like GPT/Claude are actually trained (pre-training → SFT → RLHF) — see the dedicated notes in `llm_architecture.md`.

---

## 8. Radial Basis Function Network (RBFN)

A **feedforward** network with a single hidden layer where each hidden neuron uses a **radial basis function** (typically a Gaussian) that measures the **distance** between the input and a learned "center" point, rather than the usual weighted-sum-plus-activation of a standard neuron. Neurons whose center is "close" to the input activate strongly — conceptually similar to a nearest-neighbor/clustering approach, but embedded in a trainable network.

- **Applications:** Function approximation, time-series prediction, control systems.

---

## 9. Modular Neural Network (MNN)

Instead of one large monolithic network, an MNN splits the overall problem across **several independent sub-networks (modules)**, each specializing in a portion of the task; an integrating unit then combines their outputs into the final result.

- **Why it helps:** Modules can be trained independently (even in parallel), are easier to debug and extend individually, and each module can stay simpler than one giant network trying to learn everything at once.

---

## 10. Deep Belief Network (DBN) / Boltzmann Machine

Older but historically important architectures built from stacked **Restricted Boltzmann Machines (RBMs)** — probabilistic, generative models where neurons form a fully connected, **undirected** bipartite graph (a visible layer and a hidden layer), rather than the standard directed feedforward structure. A DBN stacks multiple RBMs and trains them **layer by layer** (greedy, unsupervised pre-training) before optionally fine-tuning the whole stack for a supervised task.

- **Historical importance:** DBNs were among the first architectures to demonstrate that genuinely "deep" networks (many layers) could be trained effectively at all, helping spark the modern deep learning boom — before techniques like ReLU, better weight initialization, and batch normalization made training deep networks directly practical without this layer-wise pre-training trick.

---

## 11. Graph Neural Network (GNN)

Designed to operate directly on **graph-structured data** (nodes + edges) rather than grid-like data (images) or ordered sequences (text) — e.g., social networks, molecules, knowledge graphs. Each node repeatedly updates its own representation by **aggregating information from its neighboring nodes** (a process called **message passing**), so a node's final representation captures both its own features and the structure/context of its surrounding graph.

- **Applications:** Social network analysis, recommendation systems, drug discovery / molecule property prediction, fraud detection.

---

## 12. Siamese Network

Consists of **two (or more) identical sub-networks that share the same weights**, each processing a different input; the resulting output embeddings are then compared (typically via a distance metric like Euclidean or cosine distance) to measure how similar the two inputs are — instead of directly predicting a fixed class label.

- **Why "shared weights" matters:** It guarantees both inputs are mapped into the **same embedding space** using the exact same transformation, so the distance between their embeddings is a meaningful measure of similarity.
- **Applications:** Face verification ("are these two photos the same person?"), signature verification, one-shot learning, duplicate/similarity detection.

---

## Comparison Table: Types of Neural Networks

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

---

## Frequently Asked Questions (FAQs)

**Q1. How do I choose which architecture to use?**
Start from the shape of your data and the task: grid-like/spatial data → CNN, sequential/ordered data → RNN/LSTM/Transformer (prefer Transformer if compute allows), no labels available → Autoencoder/GAN/DBN, relational/graph data → GNN, similarity/verification task → Siamese Network, plain tabular data → FNN/MLP.

**Q2. Are these architectures mutually exclusive?**
No — real-world systems frequently combine them. For example, a video model might use a CNN to extract per-frame features and then an RNN or Transformer to model how those features change over time; a recommendation system might combine a GNN (for the user-item graph) with an autoencoder (for feature learning).

**Q3. Is a Transformer a replacement for RNNs/LSTMs?**
For most large-scale sequence tasks today (especially NLP), yes — Transformers have largely replaced RNNs/LSTMs because they train faster (parallelizable) and capture long-range dependencies better. RNNs/LSTMs still see use in smaller-scale, resource-constrained, or strictly-online/streaming settings where a Transformer's full-sequence attention isn't practical.

**Q4. Which of these are "supervised" vs "unsupervised"?**
FNN, CNN, RNN/LSTM, and Transformers are typically trained **supervised** (labeled input → output pairs) for tasks like classification/regression. Autoencoders, GANs, and DBNs are typically trained **unsupervised/self-supervised** — they learn structure directly from the data itself without needing manual labels.
