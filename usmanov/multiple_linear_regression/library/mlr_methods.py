from copy import copy
from library.matrix_methods import transpose_matrix, multiplication_matrix, inverse_matrix


def multiple_linear_regression(x: dict, y: dict) -> list:
    x_matrix = transpose_matrix([list(x[key]) for key in x])
    y_matrix = transpose_matrix([list(y[key]) for key in y])
    x_matrix = [[1] + sub_list for sub_list in x_matrix]
    b_coef = multiplication_matrix(inverse_matrix(multiplication_matrix(transpose_matrix(x_matrix), x_matrix)),
                                   multiplication_matrix(transpose_matrix(x_matrix), y_matrix))
    return b_coef


def dispersion(y: dict) -> float:
    result = 0
    for i in range(len(y['Y'])):
        result += (y['Y'][i] - sum(y['Y']) / len(y['Y'])) ** 2
    return result / (len(y['Y']) - 1)


def y_model(x: dict, y: dict, b: list) -> dict:
    b_temp = copy(b)
    b0 = b_temp.pop(0)
    b_temp = sum(b_temp, [])
    new_y = dict()
    new_y['Y'] = []
    x_matrix = transpose_matrix([list(x[key]) for key in x])

    for i in range(len(y['Y'])):
        new_y['Y'].append(sum([x * y for x, y in zip(b_temp, x_matrix[i])]) + b0[0])
    return new_y


def sse(y: dict, ymodel: dict) -> float:
    result = 0
    for i in range(len(y['Y'])):
        result += (y['Y'][i] - ymodel['Y'][i]) ** 2
    return result
