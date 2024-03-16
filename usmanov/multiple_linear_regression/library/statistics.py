def darbin_wattson(y: dict, ymodel: dict) -> float:
    e_matrix = []
    summ_1, summ_2 = 0, 0

    for i in range(len(y['Y'])):
        e_matrix.append(y['Y'][i] - ymodel['Y'][i])
        summ_2 += e_matrix[i] ** 2

    for i in range(1, len(e_matrix)):
        summ_1 += (e_matrix[i] - e_matrix[i - 1]) ** 2
    return summ_1 / summ_2