# ⚡ ZNN - Zero Neural Network

> An extensible deep learning and autograd engine built from scratch in Python.

ZNN (Zero Neural Network) is a minimal yet powerful neural network framework designed for learning, experimentation, and customization. It builds a foundation similar to PyTorch's autograd but is developed from scratch with clean architecture, sharp naming, and robust modularity.

---

## 🚀 Features

- 🔁 **Autograd Engine** — Pure Python implementation of automatic differentiation
- 🧠 **Neural Modules** — Neurons, Layers, and MLPs using `Value` nodes
- 📉 **Training Loop Templates** — Structured optimization and forward/backward passes
- 🛠️ **Extensible Design** — Easy to modify, expand, or scale
- 🎨 **Utilities & Visualization** — Clean plots and tooling for learning
- 🧪 **Test Suite** — Built-in unit tests for correctness and coverage of program

---

## 📦 Installation

### ⚡ Quick Setup (Recommended)

```bash
git clone https://github.com/samnaveenkumaroff/znn.git
cd znn
chmod +x setup.sh
./setup.sh
````

This will:

1. Create and activate a `conda` environment named `znn`
2. Install dependencies: `numpy`, `matplotlib`, `pytest`, `rich`, `jupyter`, etc.
3. Install `znn` in development mode
4. Run tests and XOR demo automatically

### 🧠 Manual Setup

```bash
conda create -n znn python=3.10 -y
conda activate znn
pip install -r requirements.txt
pip install -e .
```

---

## 🧪 Run Tests

```bash
pytest tests/
```

---

## 💡 Usage Example

```python
from znn import MLP, Value

model = MLP(2, [4, 4, 1])
x = [Value(0.5), Value(-0.2)]
y_true = Value(1.0)

y_pred = model(x)
loss = (y_pred - y_true) ** 2
loss.backward()

# Simple gradient descent
for p in model.parameters():
    p.data -= 0.01 * p.grad

print(f"Loss: {loss.data:.4f}")
```

---

## 🧠 Project Structure

```
znn/
├── znn/               # Core package
│   ├── engine.py      # Autograd engine (Value class)
│   ├── nn.py          # Neural network components
│   ├── utils.py       # Utility functions
│   └── __init__.py
├── examples/          # Usage demos (e.g. XOR, spiral)
│   ├── xor_example.py
│   └── advanced_example.py
├── tests/             # Unit tests
│   └── test_engine.py
├── requirements.txt   # Dependencies
├── setup.py           # Package setup
├── environment.yml    # Conda environment
├── setup.sh           # Auto setup script
└── README.md          # This file
```

---

## 📚 Core Concepts

### ✅ `Value` — Differentiable Scalar

```python
from znn import Value

a = Value(2.0)
b = Value(-3.0)
c = a * b + Value(1.0)
c.backward()

print(a.grad, b.grad)  # Gradients via backprop
```

### 🧠 Neural Networks

```python
from znn import Neuron, Layer, MLP

mlp = MLP(nin=2, nouts=[4, 4, 1])
output = mlp([Value(0.5), Value(-1.5)])
```

---

## 🔁 Training Loop Template

```python
for epoch in range(100):
    # Forward pass
    y_pred = model(x)
    loss = (y_pred - y_true)**2

    # Backward pass
    for p in model.parameters():
        p.grad = 0.0
    loss.backward()

    # Update
    for p in model.parameters():
        p.data -= 0.01 * p.grad
```

---

## 🎯 Demos

### XOR Training

```bash
python examples/xor_example.py
```

### Spiral Classification

```bash
python examples/advanced_example.py
```

---

## 🛡 License

Licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## Inspiration

ZNN was inspired by the clean simplicity of early neural engines. Credit to [micrograd](https://github.com/karpathy/micrograd) for sparking the seed — ZNN takes it to the next level.

---

**Made with ❤️ by [Sam Naveenkumar V](https://github.com/samnaveenkumaroff)**

> ZNN — Build it. Understand it. Evolve it.

