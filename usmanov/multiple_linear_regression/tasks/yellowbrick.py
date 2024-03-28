from yellowbrick.features import Rank2D
import pandas as pd


def pairwise_pearson_correlation(x: dict, y: dict) -> None:
    y_values = list(y.values())[0]
    x_values = list(x.values())
    df1 = pd.DataFrame(x_values)
    df1 = df1.transpose()
    df2 = pd.DataFrame(y_values)

    concatenated_columns = pd.concat([df1, df2], axis=1)

    selected_columns = ['y', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6']
    visualizer = Rank2D(features=selected_columns, algorithm='pearson')
    visualizer.fit_transform(concatenated_columns)
    visualizer.show()
