import numpy as np
from sklearn import linear_model, datasets, tree
import matplotlib.pyplot as plt

iris = datasets.load_iris()
X = iris.data[:, :2]  # Choosing only the first two input-features
Y = iris.target
# The first 50 samples are class 0 and the next 50 samples are class 1
X = X[:100]
Y = Y[:100]
number_of_samples = len(Y)
# Splitting into training, validation and test sets
random_indices = np.random.permutation(number_of_samples)
# Training set
num_training_samples = int(number_of_samples * 0.7)
x_train = X[random_indices[:num_training_samples]]
y_train = Y[random_indices[:num_training_samples]]
# Validation set
num_validation_samples = int(number_of_samples * 0.15)
x_val = X[random_indices[num_training_samples: num_training_samples + num_validation_samples]]
y_val = Y[random_indices[num_training_samples: num_training_samples + num_validation_samples]]
# Test set
num_test_samples = int(number_of_samples * 0.15)
x_test = X[random_indices[-num_test_samples:]]
y_test = Y[random_indices[-num_test_samples:]]

# Visualizing the training data
X_class0 = np.asmatrix(
    [x_train[i] for i in range(len(x_train)) if y_train[i] == 0])  # Picking only the first two classes
Y_class0 = np.zeros((X_class0.shape[0]), dtype=np.int)
X_class1 = np.asmatrix([x_train[i] for i in range(len(x_train)) if y_train[i] == 1])
Y_class1 = np.ones((X_class1.shape[0]), dtype=np.int)

X_class0 = np.array(X_class0)
X_class1 = np.array(X_class1)
plt.scatter(X_class0[:, 0], X_class0[:, 1], color='red')
plt.scatter(X_class1[:, 0], X_class1[:, 1], color='blue')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.legend(['class 0', 'class 1'])
plt.title('Fig 3: Visualization of training data')
plt.show()
