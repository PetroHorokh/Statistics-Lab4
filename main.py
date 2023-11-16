import func as f
import pandas as pd
import statsmodels.api as sm

data = pd.read_csv('data.csv')

Ni = data['Ni']
Pi = data['Pi']
Qi = data['Qi']

x = f.calculate_estimation(30, Ni, Qi)
print("Xi:")
f.array_print(x)

X = sm.add_constant(x)

Y = f.calculate_estimation(20, Ni, Pi)
print("Yi:")
f.array_print(Y)

model = f.build_regressive_model(X, Y)

f.adequacy_check(model, X)

f.regression_evaluation(model, X)

f.build_graph(model, x, X, Y)

f.prediction(model)
