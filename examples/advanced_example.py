# examples/advanced_example.py

"""Advanced Binary Classification Example using ZNN."""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
import math
from znn import Value, MLP

def generate_spiral_data(n_points=100, noise=0.1):
    """Generate spiral dataset for binary classification."""
    X, y = [], []
    
    for i in range(n_points):
        r = i / n_points
        t = 1.25 * i / n_points * 2 * math.pi
        
        # First spiral (class 0)
        x1 = r * math.cos(t) + random.gauss(0, noise)
        y1 = r * math.sin(t) + random.gauss(0, noise)
        X.append([x1, y1])
        y.append(0)
        
        # Second spiral (class 1)
        x2 = r * math.cos(t + math.pi) + random.gauss(0, noise)
        y2 = r * math.sin(t + math.pi) + random.gauss(0, noise)
        X.append([x2, y2])
        y.append(1)
    
    return X, y

def accuracy(model, X, y):
    """Calculate model accuracy."""
    correct = 0
    for xi, yi in zip(X, y):
        x_vals = [Value(x) for x in xi]
        pred = model(x_vals)
        pred_class = 1 if pred.data > 0.5 else 0
        if pred_class == yi:
            correct += 1
    return correct / len(X)

def main():
    """Train on spiral dataset."""
    print("🌀 ZNN Advanced Example: Spiral Classification")
    print("=" * 50)
    
    # Generate data
    random.seed(42)
    X_raw, y_raw = generate_spiral_data(n_points=50, noise=0.05)
    
    # Convert to Value objects
    X = [[Value(x[0]), Value(x[1])] for x in X_raw]
    y = [Value(float(yi)) for yi in y_raw]
    
    print(f"📊 Dataset: {len(X)} samples")
    print(f"🎯 Classes: {len(set(y_raw))} (binary classification)")
    
    # Create larger model for complex problem
    model = MLP(2, [16, 16, 16, 1])
    print(f"\n🧠 Model Architecture: {model}")
    print(f"🔧 Total Parameters: {len(model.parameters())}")
    
    # Training parameters
    learning_rate = 0.01
    epochs = 200
    
    print(f"\n🚀 Training Configuration:")
    print(f"   Learning Rate: {learning_rate}")
    print(f"   Epochs: {epochs}")
    print(f"   Optimizer: SGD")
    
    print("\n📈 Training Progress:")
    print("-" * 50)
    
    losses = []
    accuracies = []
    
    for epoch in range(epochs):
        # Forward pass
        ypred = [model(xi) for xi in X]
        
        # Mean squared error loss
        loss = sum((ypi - yi)**2 for ypi, yi in zip(ypred, y)) * (1.0 / len(X))
        
        # Zero gradients
        for p in model.parameters():
            p.grad = 0.0
        
        # Backward pass
        loss.backward()
        
        # Update parameters with gradient descent
        for p in model.parameters():
            p.data -= learning_rate * p.grad
        
        # Track metrics
        losses.append(loss.data)
        
        if epoch % 20 == 0 or epoch == epochs - 1:
            acc = accuracy(model, X_raw, y_raw)
            accuracies.append(acc)
            print(f"Epoch {epoch:3d}: Loss = {loss.data:.6f} | Accuracy = {acc:.3f}")
    
    print("\n🎯 Final Results:")
    print("-" * 50)
    
    final_accuracy = accuracy(model, X_raw, y_raw)
    print(f"✅ Final Accuracy: {final_accuracy:.3f} ({final_accuracy*100:.1f}%)")
    print(f"📉 Final Loss: {losses[-1]:.6f}")
    
    # Test on a few examples
    print(f"\n🔍 Sample Predictions:")
    print("-" * 30)
    for i in range(0, len(X), len(X)//5):  # Show every 5th example
        xi, yi = X[i], y[i]
        pred = model(xi)
        x_vals = [x.data for x in xi]
        pred_class = "Class 1" if pred.data > 0.5 else "Class 0" 
        actual_class = "Class 1" if yi.data > 0.5 else "Class 0"
        confidence = max(pred.data, 1 - pred.data)
        
        print(f"Input: [{x_vals[0]:6.3f}, {x_vals[1]:6.3f}] | "
              f"Predicted: {pred_class} ({confidence:.3f}) | "
              f"Actual: {actual_class}")
    
    return model, losses, accuracies

if __name__ == "__main__":
    model, losses, accuracies = main()