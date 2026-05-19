import numpy as np
import pandas as pd
from pysr import PySRRegressor

model = PySRRegressor(
    maxsize=300,
    niterations=200,  # < Increase me for better results
    binary_operators=["+", "*"],
    unary_operators=[
        "cos",
        "sin",
    ],
    # ^ Define operator for SymPy as well
    elementwise_loss="loss(prediction, target) = (prediction - target)^2",
    # ^ Custom loss function (julia syntax)
)

# data = np.genfromtxt(f'data/datasets/ttt_best_bin.csv', delimiter=',', dtype=np.int32, names=True)
# print(data.shape)

# df = pd.read_csv('data/datasets/ttt_best_bin.csv')
# data = df.to_numpy()

data = np.loadtxt('data/fixed_arr.txt')

X = data[:, :9]
y = data[:, 9:]

print(f'X.shape: {X.shape} y.shape: {y.shape}')

model.fit(X,y)
print(model)

y_pred = model.predict(X)

acc = np.sum(y[:, 0] == np.round(y_pred))/X.shape[0]
print(f'Model accuracy: {acc*100:.4f}%')