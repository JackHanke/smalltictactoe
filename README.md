# Finding a small constant time solution to Tic Tac Toe

Tic Tac Toe is a tiny, strongly solved game. There are countless tictactoe playing programs on GitHub, so why another one? Well, I want to find a *very* small constant time algorithm that is also light on memory (no hashmap). So, how small of a tic tac toe program can you make?

## Dataset(s) Creation

We can still utilize the traditional minimax algorithm to create a dataset. In fact, as there are many optimal moves in a given position, there are many datasets. 

## Representations

TODO

## Algorithm Options

| Algorithm | Rep | Result |
|-|-|-|
| [Quine McCluskey Algorithm](https://en.wikipedia.org/wiki/Quine%E2%80%93McCluskey_algorithm) | | This was too slow to complete for the dataset in question, and so is |
| [Espresso Optimizer](https://en.wikipedia.org/wiki/Espresso_heuristic_logic_minimizer) | | specifically [this implementation](https://github.com/classabbyamp/espresso-logic/tree/master) |
| Symbolic Regression, specifically using [PySR](https://github.com/MilesCranmer/PySR)| | |
| Neural Networks | Neural Net Rep | `826` parameters with a `(9,43,9)` net. |
|[Deep Differentiable Logic Gate Networks](https://github.com/Felix-Petersen/difflogic) | |

