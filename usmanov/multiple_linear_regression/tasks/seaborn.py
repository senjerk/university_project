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
    y_values = list(y.values())[0]
    x_values = list(x.values())

    df = pd.DataFrame(np.column_stack([y_values] + x_values),
                      columns=['Y'] + [f'X{i}' for i in range(1, len(x_values) + 1)])

    correlations = df.corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(correlations, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.xlabel("Features")
    plt.ylabel("Features")
    plt.show()
    interpret_correlation_heatmap(correlations)
