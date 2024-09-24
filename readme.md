# Named Entity Recognition (NER) using LSTM

This project implements a Named Entity Recognition (NER) model using a Long Short-Term Memory (LSTM) network in PyTorch. The model is trained on a large dataset of sentences and their corresponding labels, and can then be used to identify named entities in new text.

## Project Overview

The project consists of the following components:

* **Data Preprocessing:** The dataset is preprocessed to convert words and tags into numerical indices.
* **Model Architecture:** An LSTM network is defined with embedding, LSTM, and dense layers.
* **Training:** The model is trained using the Adam optimizer and cross-entropy loss.
* **Evaluation:** The model is evaluated on a separate validation and test set to measure its performance.
* **Prediction:** The trained model can be used to predict named entities in new text.

## Libraries Used

| Library | Description |
|---|---|
| PyTorch | A deep learning framework for building and training neural networks. |
| NumPy | A library for numerical computing in Python. |
| torchsummary | A library for summarizing the architecture of PyTorch models. |
| torch.nn.functional | A module for common neural network functions. |
| torch.optim | A module for optimizers used in training. |

## Usage

1. **Prepare the dataset:** Download the dataset and place it in the `data` directory.
2. **Run the Jupyter Notebook:** Open the `model.ipynb` notebook and execute the cells.
3. **Train the model:** The model will be trained on the training data.
4. **Evaluate the model:** The model will be evaluated on the validation and test data.
5. **Predict named entities:** Use the `predict` function to predict named entities in new text.

## Results

The model achieves an accuracy of over 95% on the test set.

## Future Work

* **Experiment with different model architectures:** Explore other neural network architectures, such as transformers.
* **Improve data preprocessing:** Implement more sophisticated data preprocessing techniques.
* **Fine-tune hyperparameters:** Optimize the model's hyperparameters for better performance.
