import numpy


"""

Consider an artificial neuron characterized by the parameters:
w1=1.2, w2=-0.7 w3 =0.3.
Classify the following instances
http://sebastianraschka.com/Articles/2015_singlelayer_neurons.html#frank-rosenblatts-perceptron
"""

# TODO bias
data = [
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0]
]

weights = [1.2, -0.7]
learning_rate = 0.3


def net_input(x):
    return numpy.dot(x, weights)


def predict(x):
    prediction = net_input(x)
    if prediction >= 0.0:
        return (prediction, 1)
    else:
        return (prediction, 0)

for x in data:
    print(x, predict(x))
