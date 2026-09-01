# Minimizing a simple function f(x) = x^2 using Gradient Descent
x = 10.0            # initial guess
learning_rate = 0.1
epochs = 20

for i in range(epochs):
    gradient = 2 * x          # derivative of x^2 is 2x
    x = x - learning_rate * gradient
    print(f"Epoch {i+1}: x = {x:.4f}")

# x keeps moving closer to 0, which is the minimum of f(x) = x^2