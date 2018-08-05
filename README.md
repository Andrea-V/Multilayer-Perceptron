# Andrea Valenti's Multilayer Perceptron implementation

This project provides a Python implementation of a [Multilayer Perceptron](https://en.wikipedia.org/wiki/Feedforward_neural_network#Multi-layer_perceptron).
(MLP for short) neural network.

## Features

- Simple (but inefficient) implementation of a MLP.
- Application of the MLP to the well-known [MONK dataset](https://archive.ics.uci.edu/ml/datasets/MONK%27s+Problems).

## Getting Started

- File *valentiMLP.py* contains the implementation.
- File *monk_simple.py* contains the application to the MONK tasks.

The code is ready-to-go. MONK dataset is already provided in the "dataset" subfolder.

### Installation/Dependencies

You'll need Python 3.X to run the scripts, as well as the following python libraries:
- numpy
- pandas
- matplotlib
- time
- pprint
- scikit-learn (sklearn)
- itertools
- random
- math

You can install them using your favourite packet manager for Python (such as pip).

### Usage

Just run the *monk_siple.py* script into the Python interpreter, or use the ValentiMLP class inside your own project.
You're ecouraged to explore different hyperparameters grids for the model selection, in order to do that you'll have to modify the *param_grid* variable at line 134 of *monk_simple.py*.
You can also load different MONK tasks (or different dataset) by changing line 76 and 77 of *monk_simple.py*. If you decide to use a different dataset than MONK, you will probably have to change of the preprocessing part of the script.

NB: depending on your hardware, the model selection may take a while.

## Getting Help

For any additional information about this project, you can email me at valentiandrea@rocketmail.com.

## Contributing Guidelines  

You're free (and encouraged) to make your own contributions to this project.

## Code of Conduct

Just be nice, and respecful of each other!

## License

All source code from this project is released under the MIT license.