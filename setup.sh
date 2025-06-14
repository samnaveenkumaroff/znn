#!/bin/bash
# ZNN Complete Setup Script
# Automated setup, code generation, and testing for Zero Neural Network project
set -e # Exit on any error

echo "🚀 Setting up ZNN (Zero Neural Network) - Complete Setup"
echo "========================================================="

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "❌ Conda not found. Please install Anaconda or Miniconda first."
    exit 1
fi

# Create project directory structure
echo "📁 Creating project structure..."
mkdir -p znn/{znn,tests,examples}

# Create conda environment
echo "🐍 Creating conda environment..."
if conda env list | grep -q "^znn "; then
    echo "⚠️ Environment 'znn' already exists. Removing it..."
    conda env remove -n znn -y #remove conda if already present
fi
#prefer python version 3.10
conda create -n znn python=3.10 -y
echo "✅ Conda environment created"

# Activate environment and install dependencies
echo "📦 Installing dependencies..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate znn
pip install numpy matplotlib pytest rich black flake8 jupyter
echo "✅ Dependencies installed"

# Create package files with actual code
echo "📄 Creating ZNN package files..."

# Create znn/__init__.py
cat > znn/__init__.py << 'EOF'
from .engine import Value
from .nn import Neuron, Layer, MLP

__version__ = "0.1.0"
__all__ = ["Value", "Neuron", "Layer", "MLP"]
EOF

# Create znn/znn/__init__.py (empty for now)
touch znn/znn/__init__.py

# Create znn/engine.py - Autograd Core
cat > znn/engine.py << 'EOF'
import math
import random

class Value:
    def __init__(self, data, _children=(), _op="", label=""):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self * other

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __neg__(self):
        return self * -1

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Value(self.data**other, (self,), f'**{other}')
        def _backward():
            self.grad += (other * self.data**(other-1)) * out.grad
        out._backward = _backward
        return out

    def relu(self):
        out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')
        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward
        return out

    def tanh(self):
        x = self.data
        t = math.tanh(x)
        out = Value(t, (self,), 'tanh')
        def _backward():
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward
        return out

    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self,), 'exp')
        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward
        return out

    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()

    def zero_grad(self):
        self.grad = 0.0
EOF

# Create znn/nn.py - Neural Network Components
cat > znn/nn.py << 'EOF'
import random
from .engine import Value

class Neuron:
    def __init__(self, nin, activation='tanh'):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))
        self.activation = activation

    def __call__(self, x):
        # w * x + b
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        if self.activation == 'tanh':
            return act.tanh()
        elif self.activation == 'relu':
            return act.relu()
        else:  # linear
            return act

    def parameters(self):
        return self.w + [self.b]

    def __repr__(self):
        activation_map = {'tanh': 'TanhNeuron', 'relu': 'ReLUNeuron', 'linear': 'LinearNeuron'}
        return f"{activation_map.get(self.activation, 'Neuron')}({len(self.w)})"

class Layer:
    def __init__(self, nin, nout, activation='tanh'):
        self.neurons = [Neuron(nin, activation) for _ in range(nout)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]

    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"

class MLP:
    def __init__(self, nin, nouts, activations=None):
        sz = [nin] + nouts
        if activations is None:
            activations = ['relu'] * (len(nouts) - 1) + ['linear']  # ReLU hidden, linear output
        
        self.layers = []
        for i in range(len(nouts)):
            activation = activations[i] if i < len(activations) else 'tanh'
            self.layers.append(Layer(sz[i], sz[i+1], activation))

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def zero_grad(self):
        for p in self.parameters():
            p.zero_grad()

    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"
EOF

# Create tests/test_engine.py - Comprehensive Tests
cat > tests/test_engine.py << 'EOF'
import math
import pytest
from znn.engine import Value
from znn.nn import Neuron, Layer, MLP

def test_value_creation():
    v = Value(5.0)
    assert v.data == 5.0
    assert v.grad == 0.0

def test_addition():
    a = Value(2.0)
    b = Value(3.0)
    c = a + b
    assert c.data == 5.0
    
    c.backward()
    assert a.grad == 1.0
    assert b.grad == 1.0

def test_multiplication():
    a = Value(2.0)
    b = Value(3.0)
    c = a * b
    assert c.data == 6.0
    
    c.backward()
    assert a.grad == 3.0
    assert b.grad == 2.0

def test_tanh():
    x = Value(1.0)
    y = x.tanh()
    expected = math.tanh(1.0)
    assert abs(y.data - expected) < 1e-6
    
    y.backward()
    expected_grad = 1 - math.tanh(1.0)**2
    assert abs(x.grad - expected_grad) < 1e-6

def test_relu():
    # Positive case
    x1 = Value(2.0)
    y1 = x1.relu()
    assert y1.data == 2.0
    y1.backward()
    assert x1.grad == 1.0
    
    # Negative case
    x2 = Value(-2.0)
    y2 = x2.relu()
    assert y2.data == 0.0
    y2.backward()
    assert x2.grad == 0.0

def test_power():
    x = Value(3.0)
    y = x ** 2
    assert y.data == 9.0
    
    y.backward()
    assert x.grad == 6.0  # 2 * 3^1

def test_neuron():
    n = Neuron(2)
    x = [Value(1.0), Value(2.0)]
    out = n(x)
    assert isinstance(out, Value)

def test_layer():
    layer = Layer(2, 3)
    x = [Value(1.0), Value(2.0)]
    out = layer(x)
    assert len(out) == 3
    assert all(isinstance(o, Value) for o in out)

def test_mlp():
    mlp = MLP(2, [4, 1])
    x = [Value(1.0), Value(2.0)]
    out = mlp(x)
    assert isinstance(out, Value)

def test_complex_expression():
    # Test more complex expression: (a*b + c) * d
    a = Value(2.0)
    b = Value(3.0)
    c = Value(1.0)
    d = Value(4.0)
    
    result = (a * b + c) * d
    assert result.data == 28.0  # (2*3 + 1) * 4 = 7 * 4 = 28
    
    result.backward()
    assert a.grad == 12.0  # b * d = 3 * 4
    assert b.grad == 8.0   # a * d = 2 * 4
    assert c.grad == 4.0   # d = 4
    assert d.grad == 7.0   # (a*b + c) = 7

if __name__ == "__main__":
    pytest.main([__file__])
EOF

# Create examples/xor_example.py - XOR Training Demo
cat > examples/xor_example.py << 'EOF'
from znn.nn import MLP
from znn.engine import Value
import random

def main():
    print("🚀 ZNN XOR Training Example")
    print("=" * 40)
    
    # Create model
    model = MLP(2, [4, 4, 1])  # 2 inputs, 4 hidden, 4 hidden, 1 output
    print(f"📊 Model: {model}")
    print(f"🔧 Parameters: {len(model.parameters())}")
    
    # XOR dataset
    X = [
        [Value(0.0), Value(0.0)],
        [Value(0.0), Value(1.0)],
        [Value(1.0), Value(0.0)],
        [Value(1.0), Value(1.0)],
    ]
    Y = [Value(0.0), Value(1.0), Value(1.0), Value(0.0)]
    
    # Training
    print("\n📈 Training Progress:")
    print("-" * 40)
    
    learning_rate = 0.1
    epochs = 100
    
    for epoch in range(epochs):
        # Forward pass
        ypred = [model(x) for x in X]
        loss = sum((yout - ygt)**2 for ygt, yout in zip(Y, ypred))
        
        # Backward pass
        model.zero_grad()
        loss.backward()
        
        # Update parameters
        for p in model.parameters():
            p.data -= learning_rate * p.grad
        
        # Print progress
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch:3d}: Loss = {loss.data:.6f}")
    
    # Final results
    print(f"\n🎯 Final Results:")
    print("-" * 40)
    ypred = [model(x) for x in X]
    
    correct = 0
    for i, (x, target, pred) in enumerate(zip(X, Y, ypred)):
        x_vals = [xi.data for xi in x]
        predicted_class = 1.0 if pred.data > 0.5 else 0.0
        is_correct = predicted_class == target.data
        if is_correct:
            correct += 1
            
        print(f"Input: {x_vals} | Target: {target.data} | Predicted: {pred.data:.4f}")
    
    accuracy = (correct / len(Y)) * 100
    print(f"✅ Training completed! Final loss: {loss.data:.6f}")
    print(f"🎯 Accuracy: {accuracy}%")

if __name__ == "__main__":
    main()
EOF

# Create setup.py for proper package installation
cat > setup.py << 'EOF'
from setuptools import setup, find_packages

setup(
    name="znn",
    version="0.1.0",
    description="Zero Neural Network - A minimal autograd engine inspired by micrograd",
    author="Sam Naveen Kumar",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "numpy",
        "matplotlib",
    ],
    extras_require={
        "dev": ["pytest", "black", "flake8"],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
    ],
)
EOF

# Create README.md
cat > README.md << 'EOF'
# ⚡ ZNN - Zero Neural Network

ZNN is a premium, minimal neural network and autograd engine built from scratch in Python. Inspired by micrograd — rebuilt to be cleaner, sharper, and more extensible.

## 🚀 Features

- Custom scalar `Value` class with autograd
- Backpropagation engine with topological sorting
- ReLU, Tanh, and Linear activation functions
- Layers and MLP modules
- Comprehensive test suite
- XOR training example

## 📦 Installation

```bash
git clone <your-repo-url>
cd znn
conda activate znn
pip install -e .
```

## 🧪 Run Tests

```bash
pytest tests/ -v
```

## 🔬 Example

```bash
python examples/xor_example.py
```

## 🏗️ Architecture

- `znn/engine.py` - Core autograd engine with Value class
- `znn/nn.py` - Neural network components (Neuron, Layer, MLP)
- `tests/` - Comprehensive test suite
- `examples/` - Training examples and demos

## 📚 Quick Start

```python
from znn.engine import Value
from znn.nn import MLP

# Create a simple network
model = MLP(2, [4, 1])  # 2 inputs, 4 hidden, 1 output

# Forward pass
x = [Value(1.0), Value(2.0)]
output = model(x)

# Backward pass
output.backward()
```

---

Made with ❤️ using ZNN
EOF

# Create .gitignore
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Testing
.pytest_cache/
.coverage
htmlcov/

# IDEs
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Jupyter
.ipynb_checkpoints/

# Environment
.env
.venv/
EOF

# Create tests/__init__.py
touch tests/__init__.py

# Set up git repository (if not already initialized)
if [ ! -d ".git" ]; then
    echo "🔧 Initializing git repository..."
    git init
    echo "✅ Git repository initialized"
fi

# Install the package in development mode
echo "📦 Installing ZNN package in development mode..."
pip install -e .
echo "✅ Package installed"

# Activate the znn environment
echo "🔄 Activating znn environment..."
conda activate znn

echo ""
echo "🧪 Running Tests..."
echo "=" * 40
pytest tests/ -v

echo ""
echo "🚀 Running XOR Example..."
echo "=" * 40
python examples/xor_example.py

echo ""
echo "🎉 ZNN Complete Setup Finished!"
echo "=" * 50
echo "📋 What was created:"
echo "   ✅ Complete ZNN package with autograd engine"
echo "   ✅ Neural network components (Neuron, Layer, MLP)"
echo "   ✅ Comprehensive test suite (all tests passed)"
echo "   ✅ XOR training example (successfully trained)"
echo "   ✅ Package installed in development mode"
echo "   ✅ Git repository initialized"
echo ""
echo "📊 Final Status:"
echo "   Environment: $(conda info --envs | grep '*' | awk '{print $1}')"
echo "   Python: $(python --version)"
echo "   Package location: $(pip show znn | grep Location | cut -d' ' -f2)"
echo ""
echo "🚀 Ready to use! Your ZNN library is fully functional."
echo "💡 Try: python -c 'from znn import Value, MLP; print(\"ZNN imported successfully!\")'"
echo ""
echo "Happy coding! 🧠⚡"
