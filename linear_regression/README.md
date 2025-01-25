# General Regression Problem
In statistics, a general regression problems consists in fitting a function $f: R^D -> R$ such as, for every instance $(x_n, y_n)$ of our training set:

$$
y_n = f(\textbf{x_n}) + \epsilon, \ \ \epsilon \ \tilde \ N(0, \sigma^2)
$$

where the variance $\sigma^2$ is known.

# Linear Regression
In linear regression, we will fit a function $f(\textbf{x}) = x^T \textbf{\theta} + \textbf{b} $, such that $\textbf{\theta}$ and $\textbf{b}$ are, respectively, the parameters and bias vectors. In this implementation, we will use the *maximum likelihood method*, a statistic method that consists in maximizing the probability $p(y | x)$ (more especifically, we will minimize log($p(y|x)$)). 

## Loss Function
Given that our sample is iid (independent, identically distributed), we can assume $y_n \ \tilde \ N (f(\textbf{x}), \sigma^2)$:

$$
log[p(y | x)] = log [\Pi_{i}^{N} exp(-(y - x^T \theta - b))^2 / (2 \sigma^2)] = \frac{1}{\sigma^2} \cdot \sum_ {i}^{N} (y - x^T \theta - b)^2
$$

Take the loss function $L(\textbf{\theta}, \textbf{b}) = \frac{1}{N} \cdot \sum_ {i}^{N} (y - x^T \cdot \theta - b)^2$. We must find $\theta$ and $b$ to minimize this function. This will be done using the Stochastic Gradient Descent Algorithm

## Stochastic Gradient Descent


## Overfitting and Regularization:

# Examples:


## California Housing Prices Prediction:


## Aerodynamic Properties of Rectangular Wings: