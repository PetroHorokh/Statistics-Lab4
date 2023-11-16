import random
from const import *
import scipy.stats as stats
import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np


def calculate_estimation(a, ni, i):
    data = i + (a + ni) / a * (random.uniform(0, 1) - 1 / 2)
    return list(data)


def correlation_field(x, y):
    correlation_coefficient, _ = stats.pearsonr(x, y)
    print(f"Pearson's correlation coefficient: {correlation_coefficient}")

    plt.scatter(x, y, label='Data')
    plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)), color='red', label='Line')
    plt.title('Scatter diagram and linear relationship')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()


def build_regressive_model(x, y):
    model = sm.OLS(y, x).fit()

    b0 = model.params[0]
    b1 = model.params[1]

    print("\nParameter b0:", b0)
    print("Parameter b1:", b1)

    print("Model view", f"\ny = {b0} + {b1} * x")
    print("\n", model.summary())
    return model


def adequacy_check(model, x):
    p_value_f = model.f_pvalue
    if p_value_f < alpha:
        print("regression model is adequate.")
    else:
        print("regression model is not adequate.")


def regression_evaluation(model, x):
    b0, b1 = model.params[0], model.params[1]
    std_err = model.bse
    t_stat_b0 = (b0 - 0) / std_err[0]
    p_value_b0 = 2 * (1 - stats.t.cdf(abs(t_stat_b0), len(x) - 2))
    if p_value_b0 < alpha:
        print("\nParameter b0 is significant.")
    else:
        print("Parameter b0 is not significant.")

    t_stat_b1 = (b1 - 0) / std_err[1]
    p_value_b1 = 2 * (1 - stats.t.cdf(abs(t_stat_b1), len(x) - 2))
    if p_value_b1 < alpha:
        print("Parameter b1 is significant.")
    else:
        print("Parameter b1 is not significant.")

    conf_int_b0 = model.conf_int(alpha=alpha)[0]
    conf_int_b1 = model.conf_int(alpha=alpha)[1]
    print(f"confidence interval for b0: ({conf_int_b0[0]}, {conf_int_b0[1]})")
    print(f"confidence interval for b1: ({conf_int_b1[0]}, {conf_int_b1[1]})")


def build_graph(model, x, X, Y):
    plt.scatter(x, Y, label="Data Points")
    plt.plot(x, model.predict(X), color='red', label="Regression Line")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show()


def prediction(model):
    x19 = 9.6 + 0.1 * V
    y_prediction = model.predict([1, x19])
    print("\n x19 prediction:", y_prediction[0])


def array_print(data):
    for i in range(0, len(data), 5):
        print(data[i:i + 5])
