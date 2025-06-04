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
