Project Summary
This project implements an Artificial Neural Network (ANN) to detect fraudulent credit card transactions using the widely referenced Kaggle credit card fraud detection dataset. Due to the highly imbalanced dataset with less than 0.2% fraud cases, the model addresses class imbalance using SMOTE oversampling. The ANN architecture is optimized through hyperparameter tuning with Keras Tuner’s Random Search, fine-tuning neuron counts, dropout rates, and learning rates. The final model achieves a high accuracy of 99.64%, with precision of 82.3%, recall of 80.6%, and F1-score of 81.4%, effectively balancing fraud detection with minimizing false positives.

What This Code Does-
1] Loads and preprocesses the Kaggle credit card fraud dataset, using PCA-transformed features (V1 to V28), normalized Time and Amount columns.
2] Applies SMOTE to generate synthetic samples and balance the minority fraud class.
3] Defines and trains an ANN model using TensorFlow/Keras with tunable architecture.
4] Uses Keras Tuner’s Random Search for hyperparameter optimization, searching for optimal layer sizes, dropout, and learning rates.
5] Evaluates the trained model on test data, reporting accuracy, precision, recall, and F1-score metrics.
6] Saves the best performing ANN model for future deployment.
