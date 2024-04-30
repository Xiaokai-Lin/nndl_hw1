# 3-layers Neural Network Using `numpy` from Scratch

## File structure

`\data` contains the Fashion-MNIST dataset.

`\dnn` contains the source code of the neural network.

1. `data.py` defines the function to load the dataset.
2. `neural_network_func.py` defines the fundamentals of the neural network like the linear class, activation functions, loss functions.
3. `optim.py` defines the SGD optimizer.
4. `utils.py` defines a DataLoader class to load the dataset in batches and feed it to the neural network like PyTorch's DataLoader.

`best_model` stores the best model in pkl format.

`\report_material` contains the images used in the report.

## How to train and test?

Train and test part of this project was implemented in a Jupyter Notebook file `trainer.ipynb`. The training function is also defined in the file. You can open it and **run the necessary cells** to train and test the model. Also, remember to set your `working_dir` variable existed in the first cell and second cell to the correct path.
