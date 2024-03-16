import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


def interpret_correlation_heatmap(correlation_matrix):
    for i in range(len(correlation_matrix)):
        for j in range(i+1, len(correlation_matrix)):
            correlation = correlation_matrix.iloc[i, j]
            feature1 = correlation_matrix.index[i]
            feature2 = correlation_matrix.columns[j]
            if correlation > 0.7:
                print(f"Сильная положительная корреляция между '{feature1}' и '{feature2}': {correlation}")
            elif correlation < -0.7:
                print(f"Сильная отрицательная корреляция между '{feature1}' и '{feature2}': {correlation}")
            elif correlation > 0.4:
                print(f"Умеренная положительная корреляция между '{feature1}' и '{feature2}': {correlation}")
            elif correlation < -0.4:
                print(f"Умеренная отрицательная корреляция между '{feature1}' и '{feature2}': {correlation}")
            elif correlation > 0:
                print(f"Слабая положительная корреляция между '{feature1}' и '{feature2}': {correlation}")
            elif correlation < 0:
                print(f"Слабая отрицательная корреляция между '{feature1}' и '{feature2}': {correlation}")
            else:
                print(f"Нет корреляции между '{feature1}' и '{feature2}'")


def plot_heatmap(x: dict, y: dict) -> None:
    # Преобразование словарей в списки
    y_values = list(y.values())[0]  # Предполагаем, что в словаре `y` только одно значение
    x_values = list(x.values())

    # Создание DataFrame
    df = pd.DataFrame(np.column_stack([y_values] + x_values),
                      columns=['Y'] + [f'X{i}' for i in range(1, len(x_values) + 1)])

    # Вычисление корреляций
    correlations = df.corr()

    # Построение тепловой карты
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlations, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.xlabel("Features")
    plt.ylabel("Features")
    plt.show()
    interpret_correlation_heatmap(correlations)


# def plot_heatmap(x: dict, y: dict) -> None:
#     # Преобразование словарей в списки
#     x_values = list(x.values())
#     y_values = list(y.values())[0]  # Предполагаем, что в словаре `y` только одно значение
#     feature_names = list(x.keys())
#
#     # Создание DataFrame
#     num_features = len(x_values)
#     X = np.column_stack(x_values)
#     discrete = [False] * num_features
#
#     # Добавляем фиктивный дискретный признак для целевой переменной
#     y_name = list(y.keys())[0]
#     feature_names.append(y_name)
#     X = np.column_stack((X, y_values))
#     discrete.append(True)
#
#     # Построение матрицы корреляций
#     corr_matrix = np.corrcoef(X, rowvar=False)
#
#     # Построение тепловой карты
#     plt.figure(figsize=(10, 8))
#     sns.heatmap(corr_matrix, annot=True, xticklabels=feature_names, yticklabels=feature_names, cmap="coolwarm")
#     plt.title('Pairwise Pearson Correlation')
#     plt.show()