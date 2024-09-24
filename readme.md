# Named Entity Recognition (NER) using LSTM

This repository contains a simple implementation of Named Entity Recognition (NER) using a Long Short-Term Memory (LSTM) network in PyTorch.
The model is trained on a large dataset of sentences and their corresponding labels from Kaggle, and can then be used to identify named entities in new text.

## Project Overview
The project aims to build a model that can identify and classify named entities in text, such as people, organizations, locations, and dates.

## Libraries Used

* **PyTorch:** A deep learning framework for building and training neural networks.
* **NumPy:** A library for numerical computing in Python.

## Model Architecture

The model consists of the following layers:

* **Embedding Layer**: Maps words to their corresponding vector representations.
* **LSTM Layer**: Processes the word embeddings sequentially to capture long-range dependencies.
* **Dense Layer**: Maps the LSTM output to get logits.
* **Softmax Layer**: Converts logits to probabilities.

## Results

The model achieves an accuracy of over 95% on the test set and 96% accuracy in train set.

## Usage

1. **Prepare the dataset:** Download the dataset and place it in the `data` directory.
2. **Run the Jupyter Notebook:** Open the `model.ipynb` notebook and execute the cells.
3. **Train the model:** The model will be trained on the training data.
4. **Evaluate the model:** The model will be evaluated on the validation and test data.
5. **Predict named entities:** Use the `predict` function to predict named entities in new text.

## Future Work

* **Experiment with different model architectures:** Explore other neural network architectures, such as transformers.
* **Improve data preprocessing:** Implement more sophisticated data preprocessing techniques.
* **Fine-tune hyperparameters:** Optimize the model's hyperparameters for better performance.
