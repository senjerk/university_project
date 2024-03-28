import os
from library.mlr_methods import dispersion, y_model, sse, multiple_linear_regression
from library.statistics import darbin_wattson
from tasks.yellowbrick import pairwise_pearson_correlation
from tasks.seaborn import plot_heatmap
from tasks.standart import standardize_data, linear_regression

DATA_FILE_PATH = os.path.join('data.txt')


def split_data_from_file(file_path: str) -> tuple:
    with open(file_path, 'r') as file:
        lines = file.readlines()

    y = list(map(float, lines[0].split()))
    x = {}
    for i in range(len(lines) - 1):
        x.update({f"X{i + 1}": list(map(float, lines[i + 1].split()))})

    y = {"Y": y}
    return x, y


# first string is Y list, second one are X lists
if __name__ == "__main__":
    X, Y = split_data_from_file(DATA_FILE_PATH)

    pairwise_pearson_correlation(X, Y) # task №1
    plot_heatmap(X, Y) # task №2

    X_stand, Y_stand = standardize_data(X, Y) # task №3

    b = multiple_linear_regression(X, Y)
    print(y_model(X_stand, Y_stand, b)) # task №4
    print(multiple_linear_regression(X_stand, Y_stand))
    result_data = {}
    b = multiple_linear_regression(X, Y)

    n = len(Y['Y'])
    k = len(b) - 1
    p = 1 + k
    new_Y = y_model(X, Y, b)
    sst = dispersion(Y) * (len(Y['Y']) - 1)
    ssr = dispersion(new_Y) * (len(new_Y['Y']) - 1)
    sse = sse(Y, new_Y)
    df = n - p
    r2 = ssr / sst
    r2_adj = 1 - ((1 - r2) * (n - 1)) / (n - p)
    dw = darbin_wattson(Y, new_Y)

    result_data.update(
        {'b (b0->bN)': b, 'n': n, 'k': k, 'p': p, 'sst': sst, 'ssr': ssr, 'sse': sse, 'df': df, 'r2': r2,
         'r2_adj': r2_adj, 'dw': dw})
    for key, value in result_data.items():
        print(f"{key}: {value}") # task №5
