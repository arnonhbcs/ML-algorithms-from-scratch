# General Regression Problem
In statistics, a general regression problem consists in fitting a function $f: R^D \to R$ such as, for every instance $(x_n, y_n)$ of our training set:

$$
y_n = f(\mathbf{x_n}) + \epsilon, \ \ \epsilon \ \tilde \ N(0, \sigma^2)
$$

where the variance $\sigma^2$ is known.

# Linear Regression
In linear regression, we will fit a function $f(\mathbf{x}) = x^T \mathbf{W} + \mathbf{b} $, such that $\mathbf{W}$ and $\mathbf{b}$ are, respectively, the parameters and bias vectors. In this implementation, we will use the *maximum likelihood method*, a statistical method that consists in maximizing the probability $p(y | x)$ (more specifically, we will minimize $\log(p(y|x)))$.

## Loss Function
Given that our sample is iid (independent, identically distributed), we can assume $y_n \ \tilde \ N (f(\mathbf{x}), \sigma^2)$

$$
log[p(y | x)]  = \frac{1}{\sigma^2} \cdot \sum_{i}^{N} (y - x^T W - b)^2
$$

Take the loss function $L(\mathbf{W}, \mathbf{b}) = \frac{1}{2N} \cdot \sum_ {i}^{N} (y - x^T \cdot W - b)^2$. We must find $W$ and $b$ to minimize this function. This will be done using the Gradient Descent Algorithm.

## Gradient Descent
Gradient Descent is an optimization algorithm used to minimize the loss function by iteratively updating the parameters $\mathbf{W}$ and $\mathbf{b}$ using the gradients of the loss function.

The update rules for Gradient Descent are as follows:

1. Initialize $\mathbf{W}$ and $\mathbf{b}$ randomly or with predefined values.
2. Compute the gradient of the loss function with respect to $\mathbf{W}$ and $\mathbf{b}$:

$$
\frac{\partial L}{\partial W} = -\frac{1}{N} \cdot \sum_{i=1}^N (y_i - x_i^T W - b) \cdot x_i, \quad \frac{\partial L}{\partial b} = -\frac{1}{N} \cdot \sum_{i=1}^N (y_i - x_i^T W - b)
$$

3. Update the parameters using the gradients and a learning rate $\alpha$:

$$
\mathbf{W} \leftarrow \mathbf{W} - \alpha \cdot \frac{\partial L}{\partial W}, \quad \mathbf{b} \leftarrow \mathbf{b} - \alpha \cdot \frac{\partial L}{\partial b}
$$

4. Repeat until convergence or for a predefined number of iterations.

## Overfitting and Regularization:
The maximum likelihood approach is prone to overfitting, which is when a model fits very well to the training set but generalizes poorly, resulting in bad performance with test sets. Therefore, it is useful to implement regularization methods, which consist of penalizing values for $W$ that are too big.

### L1 Regularization:
L1 regularization, also known as Lasso regularization, adds a penalty proportional to the absolute values of the parameters to the loss function. The modified loss function becomes:

$$
L(\mathbf{W}, \mathbf{b}) = \frac{1}{2N} \cdot \sum_{i}^{N} (y - x^T \cdot W - b)^2 + \lambda \cdot \sum_{j} |W_j|
$$

where $\lambda$ is a hyperparameter controlling the strength of regularization.

L1 regularization encourages sparsity in $\mathbf{W}$, leading to some coefficients being exactly zero, effectively performing feature selection. It is useful when working with high-dimensional data or when interpretability of the model is important.

### L2 Regularization:
L2 regularization, also known as Ridge regularization, adds a penalty proportional to the square of the parameters to the loss function. The modified loss function becomes:


$$
L(\mathbf{W}, \mathbf{b}) = \frac{1}{N} \cdot \sum_{i}^{2N} (y - x^T \cdot W - b)^2 + \lambda \cdot \sum_{j} W_j^2
$$

Unlike L1, L2 regularization does not promote sparsity but shrinks all parameters toward zero, helping to prevent overfitting and improve the generalization of the model. It is particularly effective when all features are relevant but need to be controlled in magnitude.


### R²: Coefficient of Determination

The R² metric, or coefficient of determination, measures how well a regression model explains the variance in the dependent variable (y). It ranges from 0 to 1, where higher values indicate better model performance. The formula is:

Where:
- $y_i$: Observed values.
- $\hat{y}_i$: Predicted values.
- $\bar{y}$: Mean of observed values.
- $N$: Number of observations.

An $R^2$ value close to 1 indicates that the model explains most of the variability in $y$.



# Example: Fuel Consumption Modeling
This dataset contains model-specific fuel consumption ratings and estimated carbon dioxide emissions for light-duty vehicles for retail sale in Canada. It has data from vehicles in the beginning of 2025. It has been made avaliable online by the Canadian Goverment.

The results are displayed in the jupyter notebook at the 'example' directory.


# References
- Linear Regression Theory: 
- - https://mml-book.github.io/
- - https://www.coursera.org/learn/machine-learning

- Dataset: https://open.canada.ca/data/en/dataset/98f1a129-f628-4ce4-b24d-6f16bf24dd64/resource/d589f2bc-9a85-4f65-be2f-20f17debfcb1

# Tools
- Python.
- Pandas.
- Numpy.
- Jupyter Notebook. 




