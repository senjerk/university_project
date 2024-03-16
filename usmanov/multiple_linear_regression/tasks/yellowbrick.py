import numpy as np
from yellowbrick.target import FeatureCorrelation


def pairwise_pearson_correlation(x: dict, y: dict) -> None:
    y_values = list(y.values())[0]
    x_values = list(x.values())

    num_features = len(x_values)
    feature_names = [f'X{i}' for i in range(1, num_features + 1)]
    X = np.column_stack(x_values)

    discrete = [False] * num_features

    y_name = list(y.keys())[0]
    feature_names.append(y_name)
    X = np.column_stack((X, y_values))
    discrete.append(True)

    visualizer = FeatureCorrelation(method='mutual_info-regression', labels=feature_names)
    visualizer.fit(X, y_values, discrete_features=discrete, random_state=0)
    visualizer.show(xlim=(-1, 1))