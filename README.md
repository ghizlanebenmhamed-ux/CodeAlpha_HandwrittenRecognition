# Handwritten Digit Recognition

A handwritten digit recognition project developed for CodeAlpha’s Machine Learning internship.

## Features

- Draw a digit and predict its value
- Load a dataset sample for prediction
- Clear the drawing area
- View model evaluation results

## Dataset and model

The project uses MNIST, a dataset of handwritten digits from 0 to 9.

A Convolutional Neural Network (CNN) is implemented with PyTorch. The trained model weights are saved in digit_cnn.pt.

## Technologies

Python, PyTorch, Tkinter, NumPy, and Pillow.

## How to run

3. Keep digit_cnn.pt and the data and results folders with the source files.
4. On Windows, double-click START_WINDOWS.bat.
5. Draw one digit and click Predict.

## Project files

- app.py: drawing interface and user controls
- model.py: CNN definition and prediction logic
- train.py: model training
- digit_cnn.pt: trained model weights
- test_project.py: project tests
- data/: dataset files
- results/: evaluation results

## Limitations

The model recognizes individual digits, not words or sentences. Mouse drawings can differ from MNIST images, so some predictions may be incorrect.
