import numpy as np
from yellowbrick.target import FeatureCorrelation


def pairwise_pearson_correlation(x: dict, y: dict) -> None:
    # Преобразование словарей в списки
    y_values = list(y.values())[0]  # Предполагаем, что в словаре `y` только одно значение
    x_values = list(x.values())

    # Создание DataFrame
    num_features = len(x_values)
    feature_names = [f'X{i}' for i in range(1, num_features + 1)]
    X = np.column_stack(x_values)

    # Создание списка дискретных признаков (все признаки непрерывные)
    discrete = [False] * num_features

    # Добавляем фиктивный дискретный признак для целевой переменной
    y_name = list(y.keys())[0]
    feature_names.append(y_name)
    X = np.column_stack((X, y_values))
    discrete.append(True)

    # Визуализация корреляции признаков с целевой переменной
    visualizer = FeatureCorrelation(method='mutual_info-regression', labels=feature_names)
    visualizer.fit(X, y_values, discrete_features=discrete, random_state=0)  # Передаем y_values
    visualizer.show(xlim=(-1, 1))