# Machine Learning: Explained in Detail

This is a full explanation of each topic: what it is, why it exists, how it works step by step, a worked example, and where it breaks. It is written in the style of tutorial websites, in plain language.

**Contents**
1. What is Machine Learning?
2. Types of Machine Learning
3. Machine Learning Workflow
4. Data Preprocessing
5. Core Concepts (loss, gradient descent, overfitting, bias-variance, regularization)
6. Algorithms in Detail
7. Ensemble Learning
8. Model Evaluation
9. Imbalanced Data
10. Feature Engineering and Selection
11. Hyperparameter Tuning
12. Neural Networks
13. Python Implementation
14. Applications, Advantages, Limitations
15. Model Deployment
16. Revision Sheet

---

## 1. What is Machine Learning?

Machine Learning (ML) is a branch of Artificial Intelligence (which is branch responsible to make system acting intelligently (fixed set of rules))  in which system learn patterns from data and make predictions or decisions without being explicitly programmed for each task.

**Traditional programming vs ML**
- Traditional programming: you give the computer **data + rules**, and it produces the **output**.
- Machine learning: you give the computer **data + output**, and it works out the **rules**.

### AI vs ML vs Deep Learning

- **AI (rule-based, no learning):**  making machines do things that normally **need human intelligence**. ML is one way of getting there, and Deep Learning is one way of doing ML. (Can be done using fixed set of rules(algos) or using ML AND DL)
- **ML:** You don't write the rules by hand. You give the computer lots of examples and it works out the pattern itself.
- **Deep Learning:** is a subset of **ML that uses multi-layer neural networks**. This is ML that uses neural networks with many layers stacked on top of each other. Each layer learns something a bit more abstract than the one before. For an image, the first layer might pick up edges, the next shapes, and the next whole objects like a face..


**Diagram:**

```
+--------------------------------------------------------------+
| AI                                                           |
|   Examples (not ML): rule-based chess program,               |
|   if-else chatbot, expert system                             |
|                                                              |
|  +--------------------------------------------------------+  |
|  | ML                                                     |  |
|  |   Examples (not DL): Linear Regression, Random Forest, |  |
|  |   K-Means, Naive Bayes, SVM, KNN                       |  |
|  |                                                        |  |
|  |  +--------------------------------------------------+  |  |
|  |  | Deep Learning                                    |  |  |
|  |  |   Examples: CNN (face recognition),              |  |  |
|  |  |   LSTM (speech), Transformers (ChatGPT)          |  |  |
|  |  +--------------------------------------------------+  |  |
|  +--------------------------------------------------------+  |
+--------------------------------------------------------------+
```
---

## 2. Types of Machine Learning

ML is divided by the kind of feedback the learner gets during training.

### 2.1 Supervised Learning

**Idea.** The algorithm learns from examples where the correct answer is already known, like a student learning with a teacher who gives the answer key.

**Data format.** Every row has input features **X** and a known output **y** (the label).

| Area (sq ft) | Bedrooms | Price (y) |
|---|---|---|
| 800 | 2 | 40 lakh |
| 1200 | 3 | 65 lakh |
| 1500 | 3 | 80 lakh |

The model learns the function `y = f(X)`. After training, you give it a new house (X) and it predicts the price (y).

**Two kinds of supervised problems:**

1. **Regression:** the output is a continuous number. Examples: house price, temperature, stock price, age.
   Algorithms: Linear Regression, Ridge, Lasso, Decision Tree Regressor, Random Forest Regressor.

2. **Classification:** the output is a category (class).
   - Binary classification: two classes (spam / not spam).
   - Multi-class classification: more than two classes (digit 0–9).
   - Multi-label classification: one input can have several labels at once (a movie tagged both "comedy" and "romance").
   Algorithms: Logistic Regression, KNN, Decision Tree, Random Forest, SVM, Naive Bayes.

**Limitation.** It needs labeled data, and labeling is often expensive (a doctor must label each X-ray).

### 2.2 Unsupervised Learning

**Idea.** There are no labels. The algorithm looks at the data and discovers structure on its own, like sorting a pile of mixed photos into groups without being told what the groups are.

**Main tasks:**

1. **Clustering:** group similar points together.
   - K-Means: partitions into K groups around centroids.
   - Hierarchical clustering: builds a tree of clusters.
   - DBSCAN: groups dense regions and marks sparse points as noise.
   - Use case: customer segmentation.

2. **Dimensionality reduction:** reduce the number of features while keeping the important information.
   - PCA, t-SNE.
   - Use case: visualizing 100-dimensional data in 2D.

3. **Association rule learning:** find items that occur together.
   - Apriori, FP-Growth.
   - Use case: "customers who buy bread also buy butter" (market basket analysis).

4. **Anomaly detection:** find rare points that differ from the rest (fraud, faulty machines).

**Limitation.** It is harder to evaluate because there is no answer key to compare against.

### 2.3 Semi-Supervised Learning

Only a small part of the data is labeled, and the rest is unlabeled. The model learns from the labeled part and uses the structure of the unlabeled part to improve.

Example: 1000 medical images, of which only 50 are labeled by doctors. The model trains on the 50, predicts labels for the remaining 950, and uses the confident predictions to train further (pseudo-labeling).

### 2.4 Reinforcement Learning

**Idea.** An **agent** learns by interacting with an **environment**. It takes an **action**, the environment returns a new **state** and a **reward** (positive or negative), and the agent tries to maximize the total reward over time. It is like training a dog with treats.

**Terms:**
- **Agent:** the learner or decision maker.
- **Environment:** the world the agent interacts with.
- **State (s):** the current situation.
- **Action (a):** what the agent can do.
- **Reward (r):** feedback after the action.
- **Policy (π):** the strategy that maps states to actions. Its the agent's strategy. It tells the agent what action to take in each state. The agent's whole job in RL is to find a good policy, one that collects the maximum total reward over time.

**Learning loop:**
```
Agent --action--> Environment
Agent <--state, reward-- Environment
```

Example: a robot learning to walk gets a positive reward for moving forward and a negative reward for falling.

Algorithms: Q-Learning, Deep Q-Network (DQN), Policy Gradient.

#### Policy in Reinforcement Learning
**Types of policy**

1. **Deterministic policy.** The same state always gives the same action: `a = π(s)`.
   Example: π(red light) = stop, and π(green light) = go.

2. **Stochastic policy.** The policy gives a probability distribution over actions, and the agent samples from it: `π(a|s) = P(taking action a when in state s)`.
   Example: in state s, π(left|s) = 0.7, π(right|s) = 0.2, π(stay|s) = 0.1. The probabilities for each state sum to 1.

Why use a stochastic policy?
- **Exploration:** the agent keeps trying other actions, so it can discover something better.
- **Unpredictability:** useful in games like rock-paper-scissors, where a predictable agent loses.
- **Partial information:** when the agent can't see the full state, randomizing can be safer.
- **Smooth training:** probabilities can be adjusted gradually with gradient methods.

**Example: grid world.** A robot is in a 3×3 grid. The goal is the top-right cell, which gives reward +10. Each step costs −1.

| State | Policy says |
|---|---|
| Cell next to goal on the left | Move right |
| Cell below the goal | Move up |
| Far bottom-left cell | Move right (or up) |

The table above is a policy. A bad policy, like "always move left", would walk into a wall and collect −1 forever. A good policy takes the shortest path to the goal.

The policy is *what to do*. The value function is *how good that is*. A value function is always defined for a particular policy, and a policy can be improved using values.

**Return and value.** The agent wants the total future reward, called the **return**: `G = r₁ + γr₂ + γ²r₃ + …`. Here γ (gamma, between 0 and 1) is the **discount factor**, which makes future rewards count a bit less than immediate ones. The value of a state under a policy is the expected return: `Vπ(s) = E[ G | start in s, follow π ]`.

**Summary**
- A policy is a mapping from states to actions, written π.
- It can be deterministic, `a = π(s)`, or stochastic, `π(a|s)`.
- The goal of RL is to find the **optimal policy π\***, the one that maximizes the expected cumulative (discounted) reward.
- Policies can be learned indirectly through value functions or directly through policy gradients, and actor-critic methods combine the two.

### 2.5 Comparison

| Type | Data | Feedback | Goal | Example |
|---|---|---|---|---|
| Supervised | Labeled | Correct answers | Predict output | House price |
| Unsupervised | Unlabeled | None | Find structure | Customer groups |
| Semi-supervised | Few labels + many unlabeled | Partial | Predict with fewer labels | Medical imaging |
| Reinforcement | None up front | Rewards | Maximize reward | Game AI |

---

## 3. Machine Learning Workflow

A real project is a pipeline of steps. Skipping any step usually hurts the final result.

**Step 1: Problem definition.** Decide what to predict, whether it is regression or classification, and what counts as success. Example: "Predict whether a customer will leave in the next 30 days. Success means recall above 80%."

**Step 2: Data collection.** Gather data from databases, files, APIs, web scraping or sensors. More relevant data usually helps more than a fancier algorithm.

**Step 3: Data preprocessing.** Clean the data: handle missing values and outliers, remove duplicates, fix wrong types, encode categories and scale features.

**Step 4: Exploratory Data Analysis (EDA).** Look at the data before modeling. Check distributions with histograms, relationships with scatter plots, correlations with a heatmap, and class balance. EDA reveals problems and gives ideas for features.

**Step 5: Feature engineering.** Create new features and select the useful ones.

**Step 6: Model selection.** Pick candidate algorithms based on the problem type, data size and need for interpretability. It is common to try a simple baseline first.

**Step 7: Training.** Fit the model on the training data. The algorithm adjusts its parameters to minimize the loss.

**Step 8: Evaluation.** Measure performance on data the model has never seen, using the right metrics.

**Step 9: Hyperparameter tuning.** Adjust the settings (learning rate, tree depth) to improve performance, using validation data or cross-validation.

**Step 10: Deployment and monitoring.** Put the model in production (an API or an app). Monitor it continuously, because real-world data changes over time. This is called **data drift** (inputs change) or **concept drift** (the relationship between inputs and output changes). The model must then be retrained.

---

## 4. Data Preprocessing

"Garbage in, garbage out." A model is only as good as its data, and a large part of real ML work is data preparation.

### 4.1 Handling Missing Values

Most algorithms cannot handle them directly.

**Techniques:**

1. **Deletion**
   - Risk: you lose information and may bias the data.
2. **Simple imputation**
   - **Mean:** for numeric columns without outliers.
   - **Median:** for numeric columns with outliers or skew (income). It is **not affected by extreme values.**
   - **Mode:** for categorical columns.
   - **Constant:** fill with "Unknown" or 0.
3. **Advanced imputation**
   - **KNN imputer:** finds the K most similar rows and fills in from their values.
   - **Iterative imputer:** models each feature using the other features.
4. **Missing indicator:** add a new column "was_missing" (0 or 1), because sometimes the fact that a value is missing is itself informative.

**Example.** Ages are [25, 30, NaN, 35, 200]. The mean (suspiciously affected by 200) would be 72.5, while the median is 32.5. The median is the safer fill.

### 4.2 Handling Outliers

An outlier is a data point that is very different from the others. It may be a genuine rare event or a data-entry error.

**Detection methods:**

1. **IQR method.**
   - Q1 is the 25th percentile and Q3 is the 75th percentile.
   - IQR = Q3 − Q1.
   - Lower bound = Q1 − 1.5 × IQR. Upper bound = Q3 + 1.5 × IQR.
   - Anything outside the bounds is an outlier.
**IQR example.** Data: 10, 12, 13, 14, 15, 16, 18, 90.
Finding Q1 and Q3
Step 1: sort and split. n = 8, so the data splits into two halves of 4.
- Lower half: 10, 12, 13, 14
- Upper half: 15, 16, 18, 90
Step 2: Q1 is the median of the lower half. With four values, the median is the average of the middle two: (12 + 13) / 2 = 12.5.
Step 3: Q3 is the median of the upper half. (16 + 18) / 2 = 17.

Q1 ≈ 12.5 and Q3 ≈ 17, so IQR = 4.5. Upper bound = 17 + 1.5 × 4.5 = 23.75. Since 90 > 23.75, it is an outlier.

2. **Z-score method.** `z = (x − μ) / σ`. If |z| > 3, the point is an outlier. It works best for roughly normal data.

3. **Visual methods.** Box plots and scatter plots.

**Treatment:**
- **Remove** if it is clearly an error.
- **Cap (winsorize):** replace values beyond a percentile (for example the 1st and 99th) with that percentile.
- **Transform:** a log transform reduces the effect of large values.
- **Keep** if it is real and important (fraud cases are outliers by nature).

### 4.3 Encoding Categorical Variables

Algorithms work with numbers, so text categories must be converted.

**Types of categorical data:**
- **Nominal:** no order (color: red, blue, green).
- **Ordinal:** has order (size: small < medium < large).

**1. Label encoding.** Assign an integer to each category: small = 0, medium = 1, large = 2.
- Good for ordinal data.
- Bad for nominal data. If red = 0, blue = 1, green = 2, the model may wrongly think green > blue > red.

**2. One-Hot encoding.** Create one binary column per category.

| Color | is_red | is_blue | is_green |
|---|---|---|---|
| red | 1 | 0 | 0 |
| blue | 0 | 1 | 0 |
| green | 0 | 0 | 1 |

- Good for nominal data, with no false order.
- Problem: a column with hundreds of categories creates hundreds of columns (high dimensionality).
- **Dummy variable trap:** one column is redundant (if the others are 0, it must be that category). For linear models, drop one column (`drop_first=True`).

**3. Target (mean) encoding.** Replace each category with the average target value for that category. This is compact for high-cardinality columns, but it can leak the target and overfit, so it must be computed using only training data (ideally with cross-validation).

**4. Frequency encoding.** Replace each category with how often it appears.

### 4.4 Feature Scaling

Features often have very different ranges: age is 0–100 while salary is 0–1,000,000. Algorithms that use distances or gradients are affected.

**Why scaling matters:**
- **Distance-based (KNN, K-Means, SVM):** a large-range feature dominates the distance.
- **Gradient descent-based (linear/logistic regression, neural networks):** unscaled features make the loss surface stretched, so convergence is slow.
- **PCA:** features with large variance dominate the components.
- **Not needed for tree-based models,** because they split on one feature at a time and only care about order.

**1. Normalization (Min-Max scaling).**
`x' = (x − min) / (max − min)`
Scales to the range [0, 1].
Example: ages range from 20 to 60. For age 30, x' = (30 − 20) / (60 − 20) = 0.25.
It is sensitive to outliers, because one extreme value squeezes the others together.

**2. Standardization (Z-score scaling).**
`x' = (x − μ) / σ`
The result has mean 0 and standard deviation 1.
Example: mean 40 and standard deviation 10. For 55, x' = (55 − 40) / 10 = 1.5.
It is less affected by outliers than Min-Max and is the usual choice for most algorithms.

**3. Robust scaling.** Uses the median and IQR: `x' = (x − median) / IQR`. It is good when there are many outliers.

**Important rule.** Compute the scaling parameters (min, max, mean, std) from the **training data only**, then apply the same values to the test data. Fitting the scaler on all data leaks information.

### 4.5 Train-Test Split and Data Leakage

The purpose of a model is to work on new, unseen data. To estimate this, we hold back part of the data.

- **Training set (70–80%):** the model learns from this.
- **Validation set:** used to tune hyperparameters and compare models.
- **Test set (20–30%):** used only once, at the end, to report final performance.

For classification, use a **stratified split** so each set keeps the same class proportions.

**Data leakage** is when information from outside the training data (usually the test set or the future) enters the model, giving unrealistically good results. Common causes:
- Scaling or imputing on the full dataset before splitting.
- Including a feature that is derived from the target.
- Random splitting of time-series data (the model sees the future). Use time-based splits instead.
- Duplicate rows ending up in both train and test.

A leaked model looks excellent during development and fails in production.

### 4.6 Cross-Validation

A single train-test split can be lucky or unlucky. Cross-validation gives a more reliable estimate.

**K-Fold CV, with K = 5:**
1. Split the data into 5 equal folds.
2. Iteration 1: train on folds 2–5, test on fold 1.
3. Iteration 2: train on folds 1, 3–5, test on fold 2.
4. Continue until each fold has been the test set once.
5. Average the 5 scores (and look at the spread too).

**Variants:**
- **Stratified K-Fold:** keeps the class ratio in each fold (use it for classification).
- **Leave-One-Out (LOOCV):** K = number of samples. Very thorough but slow.
- **Time-Series Split:** always trains on the past and tests on the future.

Common choices are K = 5 or 10.

---

## 5. Core Concepts

### 5.1 Features and Labels
- **Features (X):** the inputs the model uses to make a prediction (also called independent variables, attributes or predictors).
- **Label (y):** the value we want to predict (also called the target or dependent variable).

### 5.2 Parameters and Hyperparameters

| | Parameters | Hyperparameters |
|---|---|---|
| Set by | The algorithm, during training | You, before training |
| Examples | Weights, bias, tree split points | Learning rate, tree depth, K in KNN, number of epochs, λ |
| Learned from data? | Yes | No (tuned by trial) |

### 5.3 Loss (Cost) Function

The model needs a number that says how bad its predictions are. This is the loss. Training means finding parameters that make this number as small as possible.

Strictly, **loss** is the error on one sample, and **cost** is the average loss over the whole dataset. In practice the terms are used interchangeably.

**Regression losses:**

1. **MSE (Mean Squared Error):** `(1/n) Σ (yᵢ − ŷᵢ)²`
   - Squaring makes all errors positive and punishes large errors much more.
   - Smooth and easy to differentiate, so it works well with gradient descent.
   - Sensitive to outliers.

2. **MAE (Mean Absolute Error):** `(1/n) Σ |yᵢ − ŷᵢ|`
   - Treats all errors linearly, so it is more robust to outliers.
   - Not differentiable at 0.

3. **Huber loss:** behaves like MSE for small errors and like MAE for large ones.

**Worked example.** True values y = [3, 5, 7]. Predictions ŷ = [2, 5, 10].
Errors = [1, 0, −3].
MSE = (1 + 0 + 9) / 3 = 3.33.
MAE = (1 + 0 + 3) / 3 = 1.33.

**Classification loss:**

**Binary cross-entropy (log loss):**
`L = −[ y·log(p) + (1 − y)·log(1 − p) ]`
where p is the predicted probability of class 1.
- If the true class is 1 and the model predicts p = 0.9, loss = −log(0.9) = 0.105 (small).
- If the true class is 1 and the model predicts p = 0.1, loss = −log(0.1) = 2.303 (large).
It heavily penalizes confident wrong answers.

For multi-class problems, **categorical cross-entropy** is used.

### 5.4 Gradient Descent

Gradient descent is the main optimization algorithm. Picture standing on a foggy hill and trying to reach the valley. You cannot see the valley, but you can feel the slope under your feet. You step in the direction that goes downhill, and repeat.

- The "hill" is the loss function.
- Your position is the current parameter values.
- The slope is the **gradient** (the derivative of the loss with respect to the parameter).
- Step size is the **learning rate α**.

**Update rule:**
`θ_new = θ_old − α × (∂J/∂θ)`

The minus sign matters: the gradient points uphill, and we want to go down.

**Algorithm:**
1. Initialize parameters (randomly or zero).
2. Compute predictions and the loss.
3. Compute the gradient of the loss with respect to each parameter.
4. Update the parameters.
5. Repeat until the loss stops decreasing.

**Worked example.** Minimize J(θ) = θ². The gradient is 2θ. Start at θ = 4 with α = 0.1.

| Step | θ | Gradient (2θ) | New θ = θ − 0.1 × gradient |
|---|---|---|---|
| 1 | 4.00 | 8.00 | 3.20 |
| 2 | 3.20 | 6.40 | 2.56 |
| 3 | 2.56 | 5.12 | 2.048 |

θ moves steadily toward 0, the minimum. The steps shrink as the slope flattens.

**Effect of learning rate:**
- **Too small (0.0001):** takes forever.
- **Just right (0.1 here):** converges smoothly.
- **Too large (1.1 here):** starting at 4, θ becomes 4 − 1.1 × 8 = −4.8, then −4.8 − 1.1 × (−9.6) = 5.76, and the values keep growing. It overshoots and **diverges**.

**Variants:**

| Variant | Data per update | Pros | Cons |
|---|---|---|---|
| Batch GD | All samples | Stable, smooth path | Slow on large data, memory heavy |
| Stochastic GD (SGD) | 1 sample | Fast updates, can escape local minima | Noisy path |
| Mini-Batch GD | Small batch (32, 64, 128) | Balance of speed and stability | Needs a batch size choice |

Mini-batch is the standard in deep learning. Advanced optimizers like Momentum, RMSProp and Adam improve on plain gradient descent by adapting the step size.

**Local minima and saddle points.** For non-convex losses (neural networks), the surface has many dips. In practice, good solutions are still found, and SGD noise helps.

### 5.5 Overfitting and Underfitting

**Underfitting.** The model is too simple to capture the pattern. It performs poorly on both training and test data.
- Example: fitting a straight line to data that follows a curve.
- Cause: a too simple model, too few features, too much regularization, too little training.
- Fix: use a more complex model, add features, reduce regularization, train longer.

**Overfitting.** The model learns the training data too well, including its noise and random quirks. It performs very well on training data but poorly on new data.
- Example: a very deep decision tree that creates a leaf for every training sample.
- Cause: too complex a model, too little data, too many features, training too long.
- Fix: more data, simpler model, regularization, cross-validation, early stopping, dropout, pruning, feature selection.

**Good fit.** Performs well on both training and test data, with only a small gap.

| | Training error | Test error | Gap |
|---|---|---|---|
| Underfit | High | High | Small |
| Good fit | Low | Low | Small |
| Overfit | Very low | High | Large |

**How to diagnose:** compare training and validation scores. Learning curves (error against training size or epochs) show it: for overfitting, the validation curve starts rising while the training curve keeps falling.

### 5.6 Bias-Variance Tradeoff

- Bias and variance are the two ways a model can be wrong. Together they explain why a model does badly on data it hasn't seen.
- Bias: wrong assumptions. 
   - Bias is the error from the model being too simple or making wrong assumptions about the data. It is the gap between what the model predicts on average and the true answer.
- Variance: too sensitive to the training sample
   - Variance is how much the model changes when you train it on a different sample of data.

The tradeoff:
- A simple model has high bias and low variance. It is stable but systematically off.
- A complex model has low bias and high variance. It can fit anything, but it chases noise.

The expected prediction error of a model can be split into three parts:

`Total Error = Bias² + Variance + Irreducible Error`

Your job is to find the level of complexity where bias² plus variance is smallest

- **Bias:** the error from wrong or too simple assumptions. High-bias models miss the true pattern, which gives underfitting. Example: linear regression on curved data.
- **Variance:** the error from being too sensitive to the particular training sample. If a small change in the training data changes the model a lot, variance is high, which gives overfitting. Example: a deep unpruned decision tree.
- **Irreducible error:** noise in the data itself. No model can remove it.

**Tradeoff.** Making a model more complex lowers bias but raises variance. Making it simpler lowers variance but raises bias. The best model is at the sweet spot where total error is smallest.

| Model | Bias | Variance |
|---|---|---|
| Linear Regression | High | Low |
| Deep Decision Tree | Low | High |
| Random Forest | Low | Medium (reduced by averaging) |
| KNN with K = 1 | Low | High |
| KNN with large K | Higher | Lower |


How to tell which one you have
```
┌─────────────────────────────────────────────┬─────────────────────────────┐
│                   Symptom                   │          Diagnosis          │
├─────────────────────────────────────────────┼─────────────────────────────┤
│ Train error high, test error high (similar) │ High bias (underfitting)    │
├─────────────────────────────────────────────┼─────────────────────────────┤
│ Train error low, test error much higher     │ High variance (overfitting) │
├─────────────────────────────────────────────┼─────────────────────────────┤
│ Both low, close together                    │ Good fit                    │
└─────────────────────────────────────────────┴─────────────────────────────┘
```

#### Fixes

For high bias:
- Use a more flexible model, such as polynomial features, a deeper tree or a neural network.
- Add more or better features.
- Reduce regularization.

For high variance:
- Get more training data.
- Use a simpler model, or prune the tree.
- Add regularization (L1/L2, dropout).
- Use bagging or random forests, which average many high-variance models to cut variance.
- Use cross-validation to catch it early.

### 5.7 Regularization

Weights here is scaling the input.. 
price = w * size_of_plot
but for some small size(may be road touch) (large weight), price is higher  
Regularization discourages complex models by adding a penalty for large weights to the loss function. Large weights usually mean the model is bending itself to fit noise.

Instead of minimizing only Original Loss, the model minimizes:
`Total Loss = Prediction Error + λ * Penalty`

Here, `λ` controls how strongly we penalize large weights.
The model now has two goals:
   - Make accurate predictions.
   - Avoid unnecessarily large weights.

**L2 regularization (Ridge):**
`Loss + λ Σ wᵢ²`
- Penalizes the square of weights.
- Shrinks all weights toward zero but rarely makes them exactly zero.
- Handles multicollinearity well.

**L1 regularization (Lasso):**
`Loss + λ Σ |wᵢ|`
- Penalizes the absolute value of weights.
- Can drive some weights to exactly zero, which removes those features. So it performs automatic feature selection and gives sparse models.

**Elastic Net:** `Loss + λ₁ Σ|w| + λ₂ Σ w²`. It combines both, useful when features are correlated.

**λ (lambda, or alpha in scikit-learn)** controls strength:
- λ = 0 means no regularization.
- Small λ means weak penalty (risk of overfitting).
- Large λ means strong penalty (risk of underfitting).

Other forms of regularization: dropout and early stopping (neural networks), pruning (trees), data augmentation.

---

## 6. Algorithms in Detail

### 6.1 Linear Regression

**What it does.** Predicts a continuous value by fitting a straight line (or hyperplane) through the data.

**Equation.**
- Simple (one feature): `ŷ = w·x + b`
- Multiple: `ŷ = w₁x₁ + w₂x₂ + … + wₙxₙ + b`

where `w` is the slope/weights and `b` is the intercept/bias.

**How it learns.** It finds the `w` and `b` that minimize the MSE. This can be done in two ways:
1. **Normal equation (closed form):** `w = (XᵀX)⁻¹ Xᵀy`. Exact, but slow when there are many features.
2. **Gradient descent:** iterative. It scales to large data.

**Worked example.** Data: (x, y) = (1, 2), (2, 4), (3, 6). The best line is `y = 2x + 0`.
Try a bad line `y = 1x`: predictions = [1, 2, 3], errors = [1, 2, 3], MSE = (1 + 4 + 9)/3 = 4.67.
With `y = 2x`: predictions = [2, 4, 6], MSE = 0. Learning means moving from the first line to the second.

**Assumptions:**
1. **Linearity:** the relationship between X and y is linear.
2. **Independence:** errors are independent of each other (no autocorrelation).
3. **Homoscedasticity:** errors have constant variance across all values of X.
4. **Normality of errors:** residuals are roughly normally distributed.
5. **No multicollinearity:** the features are not strongly correlated with each other.

**Pros:** simple, fast, interpretable (each weight tells the effect of its feature).
**Cons:** cannot capture non-linear relationships, sensitive to outliers, affected by multicollinearity.

**Polynomial regression** is linear regression on features like x² and x³, which lets it fit curves.

### 6.2 Logistic Regression

**What it does.** Despite the name, it solves **classification** problems (mostly binary).

**Why not linear regression for classification?** A line can output values below 0 or above 1, which makes no sense as a probability.

**How it works.**
1. Compute a linear score: `z = w·x + b`.
2. Squash it into a probability using the **sigmoid function**: `σ(z) = 1 / (1 + e^(−z))`.

Sigmoid values:
- z = 0 gives 0.5
- z = 2 gives about 0.88
- z = −2 gives about 0.12
- Large positive z approaches 1, large negative z approaches 0.

3. Apply a threshold (default 0.5): if p ≥ 0.5, predict class 1, else class 0.

The threshold can be changed. Lowering it to 0.3 increases recall at the cost of precision.

**Training.** It minimizes binary cross-entropy (log loss) using gradient descent. There is no closed-form solution.

**Multi-class.** Use One-vs-Rest, or Softmax regression (multinomial).

**Pros:** simple, fast, gives probabilities, interpretable.
**Cons:** assumes a linear decision boundary, struggles with complex relationships.

### 6.3 K-Nearest Neighbors (KNN)

**Idea.** "You are similar to your neighbors." To classify a new point, look at the K closest training points and let them decide.

**It is a lazy learner and instance-based.** There is no training phase; it simply stores the data. All the work happens at prediction time.

**Steps for prediction:**
1. Choose K.
2. Compute the distance from the new point to every training point.
3. Pick the K nearest.
4. Classification: majority vote. Regression: average of their values.

**Distance metrics:**
- **Euclidean:** `√Σ(xᵢ − yᵢ)²` (straight-line distance, most common).
- **Manhattan:** `Σ|xᵢ − yᵢ|` (grid distance).
- **Minkowski:** a generalization of both.

**Example.** K = 3. The three nearest neighbors have classes A, A, B. The vote is 2 to 1, so the prediction is A.

**Choosing K:**
- K = 1 gives a jagged boundary and overfits (low bias, high variance).
- A very large K gives a smooth boundary but may underfit.
- Use an odd K for binary problems to avoid ties, and pick it by cross-validation.

**Requirements and issues:**
- Feature scaling is essential.
- Slow at prediction on large data, and memory heavy.
- Suffers from the curse of dimensionality, because distances become less meaningful in high dimensions.
- Sensitive to irrelevant features.

### 6.4 Decision Tree

**Idea.** Ask a series of yes/no questions about features, narrowing down to a prediction. It looks like a flowchart.

**Structure:**
- **Root node:** the first split, using the whole data.
- **Internal (decision) nodes:** tests on a feature.
- **Branches:** the outcome of the test.
- **Leaf nodes:** the final prediction.

**How the best split is chosen.** At each node, the algorithm tries splits and picks the one that makes the resulting groups as **pure** as possible (mostly one class).

**Impurity measures:**

1. **Gini impurity:** `Gini = 1 − Σ pᵢ²`
   - 0 means perfectly pure. For two classes the maximum is 0.5.
   - Example: a node with 6 "yes" and 4 "no" has p = 0.6 and 0.4. Gini = 1 − (0.36 + 0.16) = 0.48.

2. **Entropy:** `H = −Σ pᵢ log₂(pᵢ)`
   - 0 means pure. For two classes the maximum is 1.
   - Same node: H = −(0.6 × log₂0.6 + 0.4 × log₂0.4) = 0.442 + 0.529 = 0.971.

3. **Information Gain:** `IG = Entropy(parent) − Weighted average Entropy(children)`
   The split with the highest information gain is chosen.

**Worked information gain example.**
Parent: 10 samples, 5 positive and 5 negative, so entropy = 1.
Split into:
- Left: 5 samples (4 positive, 1 negative). Entropy = −(0.8 log₂0.8 + 0.2 log₂0.2) = 0.722.
- Right: 5 samples (1 positive, 4 negative). Entropy = 0.722.
Weighted child entropy = (5/10) × 0.722 + (5/10) × 0.722 = 0.722.
IG = 1 − 0.722 = **0.278**.

**For regression trees,** splits are chosen to reduce variance (MSE), and a leaf predicts the mean of its samples.

**Stopping and pruning.** Left alone, a tree grows until every leaf is pure, which overfits.
- **Pre-pruning:** limit `max_depth`, `min_samples_split`, `min_samples_leaf`.
- **Post-pruning:** grow the full tree, then cut back branches that add little value (cost-complexity pruning).

**Pros:** easy to understand and visualize, no scaling needed, handles numeric and categorical data, captures non-linear relations.
**Cons:** overfits easily, unstable (small data changes give a different tree), biased toward features with many levels.

### 6.5 Random Forest

**Idea.** A single decision tree is unstable, but many trees together are strong ("wisdom of the crowd").

**How it is built:**
1. **Bootstrap sampling:** for each tree, draw a random sample of the data with replacement (some rows repeat, some are left out).
2. **Random feature subsets:** at each split, only consider a random subset of features (commonly √n for classification).
3. Grow each tree deeply, without pruning.
4. **Aggregate:** classification by majority vote, regression by averaging.

**Why it works.** The randomness makes the trees different from each other (decorrelated). Averaging many different, high-variance trees cancels out their errors, so variance drops while bias stays low.

**Out-of-Bag (OOB) score.** About 37% of rows are left out of each bootstrap sample. Those rows can be used to test that tree, which gives a free validation estimate without a separate set.

**Feature importance.** A random forest can report how much each feature contributed to reducing impurity.

**Key hyperparameters:** `n_estimators` (number of trees), `max_depth`, `max_features`, `min_samples_leaf`.

**Pros:** accurate, robust to overfitting, handles non-linear data, little tuning needed.
**Cons:** less interpretable than one tree, slower and larger in memory.

### 6.6 Support Vector Machine (SVM)

**Idea.** Find the decision boundary (hyperplane) that separates two classes with the **largest possible margin**.

**Terms:**
- **Hyperplane:** the boundary (a line in 2D, a plane in 3D).
- **Margin:** the distance between the hyperplane and the closest points of each class.
- **Support vectors:** the closest points to the boundary. Only these points define the hyperplane. The other points could be removed without changing it.

**Why maximum margin?** A boundary far from both classes generalizes better to new points.

**Hard vs soft margin.**
- Hard margin: no misclassification allowed. Only works for perfectly separable data.
- Soft margin: allows some errors, controlled by **C**. A large C means fewer errors allowed and a narrower margin (risk of overfitting). A small C means a wider margin and more tolerance (risk of underfitting).

**Kernel trick.** For data that is not linearly separable, SVM implicitly maps it to a higher-dimensional space where a straight boundary works, without actually computing the new coordinates.
- **Linear:** for linearly separable data.
- **Polynomial:** curved boundaries.
- **RBF (Gaussian):** the most popular, very flexible. It has a parameter **gamma** (a high gamma means a very wiggly boundary and overfitting).

**Pros:** effective in high dimensions, memory efficient (uses only support vectors), flexible through kernels.
**Cons:** slow on large datasets, needs feature scaling, sensitive to hyperparameters, no direct probabilities.

### 6.7 Naive Bayes

**Foundation: Bayes' theorem.**
`P(A|B) = P(B|A) × P(A) / P(B)`

For classification: `P(class | features) ∝ P(features | class) × P(class)`

- **Prior** P(class): how common the class is.
- **Likelihood** P(features | class): how likely these features are, given the class.
- **Posterior** P(class | features): what we want.

**"Naive" assumption.** The features are independent of each other given the class, so `P(x₁, x₂, … | class) = P(x₁|class) × P(x₂|class) × …`. This is rarely true in real life, but the algorithm still often works well.

**Worked example.** 40% of emails are spam. The word "free" appears in 50% of spam emails and in 5% of non-spam emails. An email contains "free". What is the chance it is spam?

P(spam | free) = (0.5 × 0.4) / (0.5 × 0.4 + 0.05 × 0.6) = 0.20 / (0.20 + 0.03) = **0.87**.

**Types:**
- **Gaussian NB:** continuous features assumed to be normally distributed.
- **Multinomial NB:** word counts (text).
- **Bernoulli NB:** binary features (word present or not).

**Laplace smoothing.** If a word never appeared in spam in training, its probability would be 0 and wipe out the whole product. Adding 1 to all counts avoids this.

**Pros:** very fast, works with little data, good for text.
**Cons:** the independence assumption is unrealistic, and it is poor at modeling feature interactions.

### 6.8 K-Means Clustering

**Idea.** Partition the data into K clusters, where each point belongs to the cluster with the nearest center (centroid).

**Algorithm (Lloyd's):**
1. Choose K.
2. Initialize K centroids (randomly, or with k-means++ which spreads them out).
3. **Assignment step:** assign each point to its nearest centroid.
4. **Update step:** move each centroid to the mean of its assigned points.
5. Repeat 3–4 until assignments stop changing.

**Worked example.** Points on a line: 1, 2, 3, 10, 11, 12. K = 2, initial centroids 1 and 10.
- Assignment: {1, 2, 3} go to centroid 1, and {10, 11, 12} go to centroid 10.
- Update: new centroids are 2 and 11.
- Reassigning gives the same groups, so the algorithm has converged.

**Choosing K:**
- **Elbow method:** plot inertia (sum of squared distances to the nearest centroid) against K. Choose K at the "elbow", where the improvement drops sharply.
- **Silhouette score:** ranges from −1 to 1. Higher means points fit their own cluster well and fit other clusters badly.

**Limitations:**
- You must choose K in advance.
- Sensitive to the initial centroids (use k-means++ and multiple runs).
- Assumes round, similarly sized clusters.
- Sensitive to outliers and scale (scale the features first).

### 6.9 Principal Component Analysis (PCA)

**Idea.** Reduce the number of features while keeping as much information (variance) as possible, by creating new uncorrelated features called **principal components**.

**Steps:**
1. Standardize the data.
2. Compute the covariance matrix.
3. Find its eigenvectors (directions) and eigenvalues (variance along each direction).
4. Sort by eigenvalue, largest first.
5. Keep the top k components and project the data onto them.

PC1 is the direction of greatest variance, PC2 is the next greatest direction perpendicular to PC1, and so on.

**Choosing k.** Look at the **explained variance ratio** and keep enough components to retain, say, 95% of the variance.

**Uses:** visualization (reduce to 2D or 3D), faster training, removing noise and multicollinearity.
**Drawback:** the new features are combinations of the originals, so they are hard to interpret. It only captures linear relationships.

---

## 7. Ensemble Learning

**Idea.** Combine several models (weak learners) to produce a stronger one. Different models make different mistakes, and combining them cancels out errors.

### 7.1 Bagging (Bootstrap Aggregating)
- Train many models **in parallel**, each on a different bootstrap sample.
- Combine by voting or averaging.
- **Reduces variance**, so it is good for high-variance models like deep trees.
- Example: Random Forest.

### 7.2 Boosting
- Train models **sequentially**. Each new model focuses on the mistakes of the previous ones.
- **Reduces bias** (and some variance).
- **AdaBoost:** increases the weight of misclassified samples so the next learner pays more attention to them.
- **Gradient Boosting:** each new tree is trained to predict the **residual errors** of the current ensemble, using gradient descent on the loss.
- **XGBoost, LightGBM, CatBoost:** fast and powerful implementations. They dominate tabular-data competitions.
- Risk: can overfit if too many rounds, so use a small learning rate and early stopping.

### 7.3 Stacking
- Train several different models (level-0), then use their predictions as input features to a final **meta-model** (level-1) that learns how to best combine them.

### 7.4 Comparison

| | Bagging | Boosting |
|---|---|---|
| Training | Parallel | Sequential |
| Main benefit | Lowers variance | Lowers bias |
| Sampling | Bootstrap samples | Reweighted or residual-focused |
| Overfitting risk | Low | Higher |
| Example | Random Forest | XGBoost |

---

## 8. Model Evaluation

### 8.1 Regression Metrics

Using the earlier example: y = [3, 5, 7] and ŷ = [2, 5, 10].

- **MAE** = average of |errors| = (1 + 0 + 3)/3 = 1.33. Easy to interpret, in the same units as y.
- **MSE** = (1 + 0 + 9)/3 = 3.33. Punishes big errors. MSE squares the errors, so its units are squared too.
- **RMSE** = √MSE = 1.83. Same units as y, and punishes big errors. RMSE takes the square root, which brings the number back to the same units as y.
When to use which:
   - MSE is usually the loss function during training. It is smooth and easy to differentiate for gradient descent.
   - RMSE is usually the metric you report, because it is in the same units as the target and easy to interpret.
- **R² (coefficient of determination)** = `1 − SS_res / SS_tot`
  - SS_res = Σ(y − ŷ)² and SS_tot = Σ(y − ȳ)².
  - It tells what fraction of the variance in y the model explains. 1 is perfect, 0 means no better than predicting the mean, and it can be negative for a bad model.
  - **Adjusted R²** penalizes adding useless features, because plain R² never decreases when you add features.

### 8.2 Classification Metrics

**Confusion matrix.** Compares predicted and actual classes.

| | Predicted Positive | Predicted Negative |
|---|---|---|
| **Actual Positive** | TP (True Positive) | FN (False Negative) |
| **Actual Negative** | FP (False Positive) | TN (True Negative) |

- **TP:** predicted positive and it really was.
- **TN:** predicted negative and it really was.
- **FP (Type I error):** predicted positive, but it was negative (false alarm).
- **FN (Type II error):** predicted negative, but it was positive (a miss).

**Worked example.** TP = 40, FN = 10, FP = 5, TN = 45 (total 100).

| Metric | Formula | Value |
|---|---|---|
| Accuracy | (TP + TN) / Total | (40 + 45)/100 = **0.85** |
| Precision | TP / (TP + FP) | 40/45 = **0.889** |
| Recall (Sensitivity, TPR) | TP / (TP + FN) | 40/50 = **0.80** |
| Specificity (TNR) | TN / (TN + FP) | 45/50 = **0.90** |
| F1 Score | 2PR / (P + R) | 2 × 0.889 × 0.8 / 1.689 = **0.842** |
| False Positive Rate | FP / (FP + TN) | 5/50 = **0.10** |

**Meaning in plain words:**
- **Precision:** when the model says "positive", how often is it right?
- **Recall:** of all real positives, how many did the model catch?
- **F1:** a single number that balances precision and recall. It uses the harmonic mean, so it is low if either one is low.

**Why accuracy can mislead.** Suppose 990 of 1000 transactions are genuine and 10 are fraud. A model that always predicts "genuine" is 99% accurate but catches zero frauds (recall = 0). Accuracy hides this.

**Precision vs recall tradeoff.** Raising the threshold makes the model stricter: precision goes up and recall goes down. Lowering the threshold does the reverse.

**Which to prioritize:**
- **Precision** when false positives are costly. Example: marking a good email as spam, or flagging an innocent person.
- **Recall** when false negatives are costly. Example: missing a cancer case, or missing fraud.
- **F1** when both matter and classes are imbalanced.

**ROC curve and AUC.**
- The ROC curve plots **TPR (recall)** against **FPR** at every threshold.
- **AUC** is the area under it. AUC = 1.0 is perfect, 0.5 is random guessing (the diagonal line), and below 0.5 is worse than random.
- AUC can be read as "the probability that the model ranks a random positive higher than a random negative".
- On highly imbalanced data, the **Precision-Recall curve (PR-AUC)** is more informative.

**Log loss** also evaluates the quality of predicted probabilities, not just the class.

**Multi-class averaging:** macro (average per class equally), micro (pool all counts), weighted (average weighted by class size).

### 8.3 Clustering Metrics
- **Inertia:** the sum of squared distances to the cluster center (lower is better).
- **Silhouette score:** from −1 to 1, higher is better.
- **Davies-Bouldin index:** lower is better.

---

## 9. Handling Imbalanced Data

When one class is much rarer than the other (fraud, disease, defects), models tend to ignore the minority class.

**Data-level techniques:**
- **Random oversampling:** duplicate minority samples. Simple, but can overfit.
- **Random undersampling:** remove majority samples. Fast, but loses information.
- **SMOTE (Synthetic Minority Over-sampling Technique):** creates new synthetic minority points by interpolating between a minority point and one of its nearest minority neighbors.

**Algorithm-level techniques:**
- **Class weights:** give the minority class a higher weight in the loss (`class_weight='balanced'`).
- **Threshold moving:** lower the decision threshold to favor the minority class.
- Use ensemble methods designed for imbalance (balanced random forest).

**Evaluation:** use precision, recall, F1, ROC-AUC and PR-AUC rather than accuracy. Always apply resampling **only on the training data**, never on the test data.

---

## 10. Feature Engineering and Selection

### 10.1 Feature Engineering
Creating new, more informative features from existing ones. Good features often matter more than the algorithm.

- **Date features:** extract day, month, weekday, is_weekend, hour.
- **Ratios and combinations:** price per square foot, BMI = weight / height².
- **Polynomial and interaction features:** x², x₁ × x₂.
- **Binning:** convert age into groups (child, adult, senior).
- **Transformations:** log or square root to reduce skew.
- **Text features:** word counts, TF-IDF.
- **Aggregations:** a customer's average purchase in the last 30 days.

### 10.2 Feature Selection
Choosing the most useful features and dropping the rest. It reduces overfitting, speeds up training and improves interpretability.

1. **Filter methods.** Score each feature independently of any model.
   - Correlation with the target, chi-square test, ANOVA F-test, mutual information, variance threshold.
   - Fast, but ignores feature interactions.

2. **Wrapper methods.** Train models on different feature subsets and choose the best.
   - Recursive Feature Elimination (RFE), forward selection, backward elimination.
   - More accurate but computationally expensive.

3. **Embedded methods.** Selection happens during model training.
   - Lasso (L1) zeroing out weights, tree-based feature importance.

### 10.3 Curse of Dimensionality
As the number of features grows:
- The data becomes sparse in the feature space, so you need exponentially more data to cover it.
- Distances between points become similar, which hurts KNN and K-Means.
- Models overfit more easily.

Remedies: feature selection, PCA, regularization, more data.

---

## 11. Hyperparameter Tuning

Hyperparameters are not learned, so we search for good values.

1. **Grid Search.** Define a grid of values for each hyperparameter and try **every combination** with cross-validation. Example: 3 values of depth × 4 values of learning rate = 12 combinations. It is exhaustive, but the cost explodes with more parameters.

2. **Random Search.** Sample random combinations from the ranges. It often finds good settings faster than grid search, because usually only a few hyperparameters matter much.

3. **Bayesian Optimization.** Builds a model of how hyperparameters affect the score and uses past trials to choose the most promising next one. It is more efficient when each trial is expensive. (Tools: Optuna, Hyperopt.)

Always tune with cross-validation or a validation set, never on the test set.

---

## 15. Model Deployment

Training a model in a notebook is only half the job. Deployment means putting the model somewhere real users or other systems can send it data and get predictions back, and keeping it working after that.

### 15.1 Saving the Model

First the trained model has to be saved to disk so it can be loaded later without retraining.

- **pickle / joblib** for scikit-learn models. joblib is better for big NumPy arrays.
- **`model.save()`** for Keras/TensorFlow, **`torch.save(state_dict)`** for PyTorch.
- **ONNX** is a framework-neutral format. Train in PyTorch, serve anywhere.

Save the whole preprocessing + model together (a scikit-learn `Pipeline`), not just the model. If the scaler or encoder is not saved with it, the live data will be transformed differently from the training data and the predictions will be wrong with no error message.

```python
import joblib
joblib.dump(pipeline, "model_v1.joblib")
pipeline = joblib.load("model_v1.joblib")
```

Only load pickle files you trust. Unpickling can run arbitrary code.

### 15.2 Ways to Serve Predictions

| Style | How it works | Good for |
|---|---|---|
| **Batch** | Run the model on a big file/table on a schedule (nightly) and store the results | Recommendations, churn scores, reports. Latency does not matter |
| **Online (real-time)** | Model sits behind an API, each request gets a prediction in milliseconds | Fraud check, search ranking, chatbots |
| **Streaming** | Model reads from a stream (Kafka) and scores events as they arrive | Sensor data, live monitoring |
| **Edge / on-device** | Model runs on the phone or device itself (TFLite, Core ML, ONNX Runtime) | Offline use, privacy, very low latency |

### 15.3 Building a Prediction API

The common approach is to wrap the model in a small web service. Load the model once when the server starts, not on every request.

```python
from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()
model = joblib.load("model_v1.joblib")

class Features(BaseModel):
    age: float
    income: float
    tenure: int

@app.post("/predict")
def predict(f: Features):
    X = [[f.age, f.income, f.tenure]]
    return {"prediction": int(model.predict(X)[0])}
```

Frameworks: **Flask** (simple), **FastAPI** (fast, validates input automatically), **TensorFlow Serving / TorchServe / BentoML** (built for models). Validate the input. Users will send missing fields, wrong types and strange values.

### 15.4 Packaging with Docker

"It works on my machine" is the usual deployment failure: different Python version, different library version. Docker packages the code, the model file, and the exact dependencies into one image that runs the same everywhere.

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Pin library versions in `requirements.txt`. A model pickled with one scikit-learn version may not load in another.

### 15.5 Where to Host It

- **Cloud ML platforms:** AWS SageMaker, Google Vertex AI, Azure ML. They handle scaling, endpoints and monitoring for you.
- **Containers:** push the Docker image to a service like AWS ECS/Fargate, Cloud Run, or Kubernetes when you need more control.
- **Serverless** (AWS Lambda): good for light models and spiky traffic, but cold starts add delay and size limits apply.
- **Simple demos:** Streamlit, Gradio, Hugging Face Spaces.

Scaling: run several copies behind a load balancer, and add autoscaling so copies go up and down with traffic.

### 15.6 CI/CD and Versioning for ML

- Version the **code** (Git), the **data** (DVC, lakeFS) and the **model** (MLflow Model Registry, W&B). You should be able to say which data and code produced the model running in production.
- A CI/CD pipeline runs tests, builds the Docker image and deploys it automatically when code changes. For ML it can also retrain and check the new model's metrics before it is allowed to ship.
- **MLOps** is the name for this whole practice: treating models like software that needs testing, versioning, deployment and monitoring.

### 15.7 Safe Rollout Strategies

Never swap a new model in for everyone at once.

- **Shadow deployment.** New model gets the live traffic and makes predictions, but nobody sees them. Compare against the old model quietly.
- **Canary release.** Send a small share (say 5%) of traffic to the new model. If metrics hold, increase it.
- **A/B test.** Split users between old and new and measure a real business metric (clicks, conversions), not just accuracy.
- **Rollback.** Keep the previous version ready so you can switch back in minutes.

### 15.8 Monitoring After Deployment

A model starts getting worse the day it ships, because the world keeps changing. Things to watch:

- **Service health:** latency, error rate, throughput, memory/CPU.
- **Data drift:** the input distribution moves away from training data (average income changes, a new product category appears). Check with PSI, KS test, or simple summary statistics.
- **Concept drift:** the relationship between inputs and the answer changes (customers behave differently after a price change), so even the same inputs now need different predictions.

Data drift is when the inputs change but the underlying rules stay the same. Say you trained a loan model on customers earning ₹30-60k a month, and now most applicants earn ₹80k+. The model has barely seen people like that, so its predictions get shaky. Nothing about how income relates to repayment changed. The model is just being asked about data it didn't learn from. You catch this by comparing new input data to the training data. PSI and the KS test do that formally, or you can just compare means, ranges, and category counts.

Concept drift is when the inputs look the same but the right answer is different now. Say you predict who will buy, then you raise prices. The same customer profile, with the same age, location, and browsing history, now buys less often. The input data looks unchanged, so input checks won't flag it. The only way to catch it is by watching the model's actual performance (accuracy, error rate) once the true outcomes come in.

- Data drift shows up in the inputs right away, and you don't need true labels for it. You compare today's feature distributions against training (PSI, KS test, mean/null-rate checks) and alert when they move.
- Concept drift doesn't show up in the inputs at all. You need real outcomes (did the customer actually buy or default?) and you track accuracy, precision/recall, or error over time. The catch is that labels often arrive late, which is why data drift checks work as the early warning.

- **Model performance:** compare predictions to the true outcomes once those become known. Labels often arrive late (a loan default shows up months later), so use drift in inputs and predictions as an early warning.
- **Training-serving skew:** features computed one way in training and another way in production. A frequent hidden bug. A feature store helps by sharing one definition.

When drift or a performance drop is found, **retrain** on fresh data, either on a schedule or triggered by an alert.

### 15.9 Practical Concerns

- **Latency:** big models are slow. Options are a smaller model, quantization (lower precision numbers), pruning, distillation, batching requests, or GPU serving.
- **Cost:** a GPU endpoint left running all day is expensive. Scale to zero when idle if you can.
- **Security and privacy:** authenticate the API, rate-limit it, do not log personal data, and watch for people probing the model.
- **Fairness and explainability:** log inputs and outputs for auditing, and be able to explain a decision (SHAP, LIME) when the use case needs it.

### 15.10 Deployment Checklist

1. Save the full pipeline, not only the model.
2. Wrap it in an API with input validation.
3. Put it in Docker with pinned dependencies.
4. Deploy to a cloud endpoint or container service.
5. Roll out slowly (shadow, canary or A/B), with rollback ready.
6. Monitor health, drift and accuracy.
7. Retrain and redeploy when things degrade.

---

## 16. Revision Sheet

| Topic | Remember |
|---|---|
| ML definition | Learns rules from data + answers (E, T, P) |
| Supervised | Labeled data: regression (number), classification (category) |
| Unsupervised | No labels: clustering, dimensionality reduction, association |
| Reinforcement | Agent, environment, reward, policy |
| Leakage | Never let test information reach training; split first, then fit scalers/imputers on train only |
| Scaling | Needed for KNN, SVM, K-Means, PCA, gradient-based models; not for trees |
| Gradient descent | `θ = θ − α·∂J/∂θ`; learning rate too high diverges, too low is slow |
| Underfit / Overfit | High bias / high variance |
| Total error | Bias² + Variance + Irreducible error |
| L1 vs L2 | L1 gives zeros (feature selection), L2 shrinks evenly |
| Gini / Entropy | Impurity of a node; pick the split with highest information gain |
| Bagging vs Boosting | Parallel, lowers variance / Sequential, lowers bias |
| Precision | TP / (TP + FP) |
| Recall | TP / (TP + FN) |
| F1 | Harmonic mean of precision and recall |
| Imbalanced data | Do not trust accuracy; use F1, PR-AUC, SMOTE, class weights |
| AUC | 0.5 random, 1.0 perfect |
| ReLU | max(0, z), the default hidden activation |
| Backpropagation | Chain rule to compute gradients of loss w.r.t. weights |
