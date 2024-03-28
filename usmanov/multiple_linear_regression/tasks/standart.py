import pandas as pd
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from yellowbrick.features import FeatureImportances
import numpy as np
import matplotlib.pyplot as plt

def linear_regression(x, y):
    # Преобразуем входные данные в массивы NumPy
    X = np.array([x[key] for key in x.keys()]).T
    y = np.array(y['Y'])

    # Разделим данные на обучающий и тестовый наборы
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Создадим и обучим модель линейной регрессии
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Визуализируем важность факторов
    plt.figure(figsize=(10, 6))
    viz = FeatureImportances(model, labels=list(x.keys()), relative=False)
    viz.fit(X, y)
    viz.poof()


def standardize_data(x: dict, y: dict):
    cols = list(x.keys())

    sca = preprocessing.StandardScaler()

    x_values = np.array([list(x[key]) for key in cols]).T
    y_values = np.array(list(y.values())).reshape(-1, 1)

    x_standardized = sca.fit_transform(x_values)
    x_standardized = pd.DataFrame(x_standardized, columns=cols)

    y_standardized = sca.fit_transform(y_values)
    y_standardized = pd.DataFrame(y_standardized, columns=['Y'])

    return x_standardized, y_standardized