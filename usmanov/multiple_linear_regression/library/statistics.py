import numpy as np
from scipy.stats import t


def darbin_wattson(y: dict, ymodel: dict) -> float:
    e_matrix = []
    summ_1, summ_2 = 0, 0

    for i in range(len(y['Y'])):
        e_matrix.append(y['Y'][i] - ymodel['Y'][i])
        summ_2 += e_matrix[i] ** 2

    for i in range(1, len(e_matrix)):
        summ_1 += (e_matrix[i] - e_matrix[i - 1]) ** 2
    return summ_1 / summ_2


def tstatistic(X: dict, Y: dict) -> bool:
    X_matrix = np.column_stack([np.array(values) for values in X.values()])
    Y_array = np.array(Y["Y"])
    X_matrix = np.column_stack((np.ones(len(X_matrix)), X_matrix))
    b = np.linalg.inv(X_matrix.T @ X_matrix) @ X_matrix.T @ Y_array
    residuals = Y_array - X_matrix @ b
    n = len(Y_array)
    k = X_matrix.shape[1] - 1
    df = n - k - 1
    mse = np.sum(residuals ** 2) / df
    se = np.sqrt(np.diag(np.linalg.inv(X_matrix.T @ X_matrix) * mse))
    t_stats = b / se
    p_values = [2 * (1 - t.cdf(np.abs(t_stat), df)) for t_stat in t_stats]
    for i, (key, t_stat, p_value) in enumerate(zip(["b0"] + list(X.keys()), t_stats, p_values)):
        if i != 0:
            if np.abs(t_stat) < 1:
                print(f"{key} статистически неважен")
            elif 1 <= np.abs(t_stat) < 2:
                print(f"{key} относительно значим")
            else:
                print(f"{key} очень важен")
    if np.abs(t_stats[0]) < 1:
        return False
    else:
        return True
