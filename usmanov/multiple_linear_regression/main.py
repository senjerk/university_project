import numpy as np


def multiple_linear_regression(X, Y):
    X_matrix = np.array([list(X[key]) for key in X]).T
    Y_matrix = np.array(list(Y.values())).reshape(-1, 1)

    X_with_intercept = np.concatenate((np.ones((X_matrix.shape[0], 1)), X_matrix), axis=1)

    coefficients = np.linalg.pinv(X_with_intercept) @ Y_matrix

    b0 = coefficients[0][0]
    b = coefficients[1:]

    return b0, b


def split_data(data):
    lines = data.split('\n')
    X1 = list(map(float, lines[0].split()))
    X2 = list(map(float, lines[1].split()))
    Y = list(map(float, lines[2].split()))

    X = {"X1": X1,
         "X2": X2}
    Y = {"Y": Y}
    return X, Y


data = """2400	2450	2450	2500	2500	2500	2700	2700	2700	2750	2775	2800	2800	2900	2900	3000	3075	3100	3150	3200	3200	3200	3225	3250	3250	3250	3500	3500	3500	3600	3900
54.5	56	58.5	43	58	59	52.5	65.5	68	45	45.5	48	63	58.5	64.5	66	57	57.5	64	57	64	69	68	62	64.5	48	60	59	58	58	61
60	61	65	30.5	63.5	65	44	52	54.5	30	26	23	54	36	53.5	57	33.5	34	44	33	39	53	38.5	39.5	36	8.5	30	29	26.5	24.5	26.5
"""

X, Y = split_data(data)

b0, b = multiple_linear_regression(X, Y)

print("b2", b[1])
print("b1", b[0])
print("b0:", b0)
