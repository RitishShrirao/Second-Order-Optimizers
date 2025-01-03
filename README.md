# Shampoo, Natural Gradient Descent implementation project

This project implements and compares the performance of three optimization algorithms: Shampoo, Natural Gradient Descent (NGD), and Adam. The implementation is focused on training a model on the CIFAR-10 dataset.

## Project Structure

*   `benchmark_cifar10.ipynb`: A Jupyter notebook that trains a model on CIFAR-10 using Shampoo, NGD, and Adam optimizers, allowing for direct performance comparison.
*   `visualization.ipynb`:  A Jupyter notebook that generates visualizations of different optimizers to aid understanding.
*   `requirements.txt`: Lists the Python packages required to run the project.
*   `natural_grad.py` : Implementation of natural gradient descent optimizer.
*   `shampoo.py`: Implementatoin of shampoo algorithm.
*   `autograd.ipynb`: Implementation of autograd and the above optimization algorithms in it.
*   `README.md`: This file.

## Getting Started

1.  **Navigate to the project directory:**
    Open your terminal or command prompt and navigate to the directory containing the project files.

2.  **Install requirements:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the benchmark:**
    Open and execute the `benchmark_cifar10.ipynb` notebook. The notebook will train the model with the different optimizers and output their performance.

4.  **Explore the visualizations:**
    Open and execute the `visualization.ipynb` notebook to see graphical representations of how the optimizers behave.

5.  **Run the above algorithms in autograd:**
    Open and execute `autograd.ipynb` notebook to run the implementation of above algorithms in an autograd engine. We've also added support for finding hessian using the autograd.

## Requirements

The project relies on the following Python packages (see `requirements.txt` for the specific versions):

*   `torch`
*   `torchvision`
*   `numpy`
*   `matplotlib`
*   `jupyter`

## Usage

The primary entry points for the project are the two Jupyter notebooks. Use them to train the model, compare the optimizers, and view visualizations of their behavior.
