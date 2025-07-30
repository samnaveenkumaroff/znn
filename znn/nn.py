import random
from .engine import Value

#Neuron init
class Neuron:
    def __init__(self, nin, activation='tanh'):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))
        self.activation = activation

    def __call__(self, x):
        #  linear function formula w * x + b 
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
