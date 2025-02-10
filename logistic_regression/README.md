# Classification Problems

In classification problems, our objective is to assign each instance to one of a set of discrete classes. Unlike regression tasks, where the output is continuous, classification involves predicting categorical outcomes (e.g., 0 or 1 in binary classification). Logistic regression is a popular method for such tasks, as it provides probabilistic outputs that can be thresholded to make final class decisions.

# Logistic Regression

Logistic regression models the probability that a given instance belongs to a specific class using the logistic (sigmoid) function. The model computes a linear combination of input features and passes it through the sigmoid function, ensuring the output is between 0 and 1. Mathematically, this is expressed as:

$$
p(y=1|\mathbf{x}) = \sigma(\mathbf{x}^T\mathbf{W} + b) = \frac{1}{1+e^{-(\mathbf{x}^T\mathbf{W} + b)}}
$$

where \( \mathbf{W} \) and \( b \) represent the model's weights and bias, respectively. The resulting probability is then used to classify the instance, typically using a threshold of 0.5.

## Loss Function

The loss function in logistic regression is derived from the likelihood of the observed data and is commonly known as the cross-entropy loss or log-loss. For a binary classification problem with \( N \) samples, the loss function is defined as:

$$
L(\mathbf{W}, b) = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i \log(p_i) + (1-y_i) \log(1-p_i) \right]
$$

where $p_i = \sigma(\mathbf{x}_i^T\mathbf{W} + b)$ is the predicted probability for sample  $i$, and  $y_i$ is the true binary label. This convex loss function allows efficient optimization via Gradient Descent.

## Gradient Descent

Gradient Descent is used to minimize the cross-entropy loss by iteratively updating the model parameters \( \mathbf{W} \) and \( b \). The gradients of the loss with respect to these parameters are given by:

$ \frac{\partial L}{\partial \mathbf{W}} = \frac{1}{N} \sum_{i=1}^{N} \left( p_i - y_i \right) \mathbf{x}_i, \quad \frac{\partial L}{\partial b} = \frac{1}{N} \sum_{i=1}^{N} \left( p_i - y_i \right) $

The parameters are then updated using the learning rate \( \alpha \):

$$
\mathbf{W} \leftarrow \mathbf{W} - \alpha \cdot \frac{\partial L}{\partial \mathbf{W}}, \quad b \leftarrow b - \alpha \cdot \frac{\partial L}{\partial b}
$$

These steps are repeated until the loss converges or the maximum number of iterations is reached.

## Overfitting and Regularization

Logistic regression, like other models, can overfit the training data, especially when dealing with high-dimensional or noisy datasets. Regularization techniques are employed to mitigate overfitting by penalizing large weights:

- **L1 Regularization (Lasso):**  
  Adds a penalty proportional to the absolute values of the weights:
  
  $ L(\mathbf{W}, b) = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i \log(p_i) + (1-y_i) \log(1-p_i) \right] + \lambda \sum_{j} |W_j| $
  
  This technique encourages sparsity in the model parameters, effectively performing feature selection.

- **L2 Regularization (Ridge):**  
  Adds a penalty proportional to the square of the weights:
  
  $ L(\mathbf{W}, b) = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i \log(p_i) + (1-y_i) \log(1-p_i) \right] + \lambda \sum_{j} W_j^2 $
  
  L2 regularization shrinks the weights towards zero, reducing model complexity without enforcing sparsity.

## Performance Measures

To evaluate a logistic regression model, several performance metrics are commonly used:

- **Accuracy:** The proportion of correctly classified instances among all instances.
- **Precision:** The ratio of true positive predictions to the total number of positive predictions, indicating the model's ability to avoid false positives.
- **Recall (Sensitivity):** The ratio of true positive predictions to the total number of actual positives, measuring the model's ability to capture all positive instances.
- **F1-score:** The harmonic mean of precision and recall, balancing the trade-off between the two metrics.

These metrics provide a comprehensive evaluation of the classifier’s strengths and limitations, particularly in the presence of imbalanced datasets.

# Example

In this implementation, we use the Breast Cancer dataset from scikit-learn, which contains features computed from breast tissue images and binary labels indicating whether a tumor is malignant or benign. The accompanying Jupyter Notebook demonstrates:
- Loading and preprocessing the dataset.
- Training a logistic regression model using gradient descent.
- Evaluating the model using accuracy, precision, recall and F1-score.
- Applying regularization techniques to prevent overfitting.

# References
- Logistic Regression Theory: [Coursera - Machine Learning](https://www.coursera.org/learn/machine-learning)
- Dataset: [Kaggle Titanic Dataset](https://www.kaggle.com/c/titanic/data)
- Model Performance Evaluation: [KDnuggets - Classification Metrics](https://www.kdnuggets.com/2022/10/classification-metrics-walkthrough-logistic-regression-accuracy-precision-recall-roc.html)

# Tools
- Pandas
- Numpy
- Jupyter Notebook
