import numpy

learning_rate = 0.25
epochs = 4
weights = numpy.array([0, -0.3, 0.0, 0.4])


x = numpy.array([
    [1, 0, 0],
    [1, 0, 1],
    [1, 1, 0],
    [1, 1, 1]
])

y = numpy.array([0, 0, 0, 1])


def net_input(X):
    # weights * x + bias
    return numpy.dot(X, weights[1:]) + weights[0]


def predict(x):
    prediction = net_input(x)
    if prediction >= 0.0:
        return prediction
    else:
        return prediction


errors_ = []
for _ in range(epochs):
    errors = 0
    for xi, target in zip(x, y):
        print('\nObject:', xi, 'Target:', target)
        print('Initial weights:', weights[1:])
        update = learning_rate * (target - predict(xi))
        print('Error:', target - predict(xi))
        weights[1:] +=  update * xi
        print('New weights:', weights[1:])
        weights[0] +=  update
        errors += int(update != 0.0)
        errors_.append(errors)
    print('*' * 100)
