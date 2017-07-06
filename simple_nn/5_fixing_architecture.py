from scipy.stats import logistic


"""
X1  X2  Y   
0   O   0  
0   1   1
1   0   1
1   1   0
1   0.5 0


w_h1    wh2     w_out_layer
20      -20         20
20      20          20

b_h1    b_h2    b_output_layer
-10     30          -35

"""

x = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
    [1, 0.5]
]

y = [0, 1, 1, 0, 0]

weights = [
    [20, 20, -10],
    [-20, -20, 30],
    [20, 20, -35]
]


def hidden_neuron_activation(x, weight):
    activation = (x[0] * weight[0]) + (x[1] * weight[1]) + weight[-1]
    print(x, weight, activation, logistic.cdf(activation), round((logistic.cdf(activation)), 1))
    return round((logistic.cdf(activation)), 1)


def output_activation(h1, h2, weight, y):
    prediction = (h1 * weight[0]) + (h2 * weight[1]) + weight[-1]
    prediction = round((logistic.cdf(prediction)), 1)
    return prediction


for index in range(0, len(y)):
    pr = output_activation(
        hidden_neuron_activation(x[index], weights[0]),
        hidden_neuron_activation(x[index], weights[1]),
        weights[2], y[index]
    )
    print('Object:', x[index], 'Target:', y[index], 'Prediction:', pr, 'Checking:', y[index] == pr, '\n')




