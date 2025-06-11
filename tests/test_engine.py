#Test Model
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
