# znn/utils.py
#simplified
import matplotlib.pyplot as plt
import numpy as np
from .engine import Value

def draw_dot(root, format='svg', rankdir='LR'):
    """
    Build a visual representation of a computational graph.
    format: 'svg' | 'png' | 'pdf'
    rankdir: 'TB' (top to bottom graph) | 'LR' (left to right)
    """
    assert rankdir in ['LR', 'TB']
    
    nodes, edges = trace(root)
    dot = Digraph(format=format, graph_attr={'rankdir': rankdir})
    
    for n in nodes:
        dot.node(name=str(id(n)), label="{ %s | data %.4f | grad %.4f }" % (n.label, n.data, n.grad), shape='record')
        if n._op:
            dot.node(name=str(id(n)) + n._op, label=n._op)
            dot.edge(str(id(n)) + n._op, str(id(n)))
    
    for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)
    
    return dot

def trace(root):
    """Build a set of all nodes and edges in a graph."""
    nodes, edges = set(), set()
    
    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)
    
    build(root)
    return nodes, edges

def plot_loss(losses, title="Training Loss"):
    """Plot training loss over epochs."""
    plt.figure(figsize=(10, 6))
    plt.plot(losses, 'b-', linewidth=2)
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    plt.show()

def generate_data(n_samples=100, noise=0.1):
    """Generate some sample data for testing."""
    np.random.seed(42)
    X = np.random.randn(n_samples, 2)
    # XOR-like pattern
    y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(float)
    # Add noise
    y = y + noise * np.random.randn(n_samples)
    return X, y

def plot_decision_boundary(model, X, y, title="Decision Boundary"):
    """Plot model decision boundary with data points."""
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # Make predictions on meshgrid
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    Z = []
    for point in mesh_points:
        pred = model([Value(point[0]), Value(point[1])])
        Z.append(pred.data if hasattr(pred, 'data') else pred[0].data)
    
    Z = np.array(Z).reshape(xx.shape)
    
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, levels=50, alpha=0.8, cmap=plt.cm.RdYlBu)
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu, edgecolors='black')
    plt.colorbar(scatter)
    plt.title(title)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()
