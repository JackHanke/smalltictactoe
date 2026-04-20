import numpy as np
import pandas as pd
from pysr import PySRRegressor

model = PySRRegressor(
    maxsize=100,
    niterations=100,  # < Increase me for better results
    binary_operators=["+", "*"],

    # ^ Define operator for SymPy as well
    elementwise_loss="loss(prediction, target) = (prediction - target)^2",
    # ^ Custom loss function (julia syntax)
)

# data = np.genfromtxt(f'data/datasets/ttt_best_bin.csv', delimiter=',', dtype=np.int32, names=True)
# print(data.shape)

df = pd.read_csv('data/datasets/ttt_best_bin.csv')
data = df.to_numpy()

X = data[:, :17]
y = data[:, 17:19]

print(f'X.shape: {X.shape} y.shape: {y.shape}')

model.fit(X,y)
print(model)