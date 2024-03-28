from yellowbrick.features import Rank2D
import pandas as pd


def pairwise_pearson_correlation(x: dict, y: dict) -> None:
    x_values = pd.DataFrame(x)
    x_values.columns = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6']

    selected_columns = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6']
    visualizer = Rank2D(features=selected_columns, algorithm='pearson')
    visualizer.fit_transform(x_values)
    visualizer.show()
