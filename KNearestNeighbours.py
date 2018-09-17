from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
from sklearn import neighbors
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris()
X = iris.data[:, :2]  # Choosing only the first two input-features
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Visualizing the training data
X_class0 = np.asmatrix(
    [X_train[i] for i in range(len(X_train)) if y_train[i] == 0])  # Picking only the first two classes
Y_class0 = np.zeros((X_class0.shape[0]), dtype=np.int)
X_class1 = np.asmatrix([X_train[i] for i in range(len(X_train)) if y_train[i] == 1])
Y_class1 = np.ones((X_class1.shape[0]), dtype=np.int)
X_class2 = np.asmatrix([X_train[i] for i in range(len(X_train)) if y_train[i] == 2])
Y_class2 = np.full((X_class2.shape[0]), fill_value=2, dtype=np.int)

X_class0 = np.array(X_class0)
X_class1 = np.array(X_class1)
X_class2 = np.array(X_class2)
plt.scatter(X_class0[:, 0], X_class0[:, 1], color='red')
plt.scatter(X_class1[:, 0], X_class1[:, 1], color='blue')
plt.scatter(X_class2[:, 0], X_class2[:, 1], color='green')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.legend(['class 0', 'class 1', 'class 2'])
plt.title('Fig 3: Visualization of training data')
plt.show()

model = neighbors.KNeighborsClassifier(n_neighbors=5)  # K = 5
model.fit(X_train, y_train)

query_point = np.array([5.9, 2.9])
true_class_of_query_point = 1
predicted_class_for_query_point = model.predict([query_point])
print("Query point: {}".format(query_point))
print("True class of query point: {}".format(true_class_of_query_point))

neighbors_object = neighbors.NearestNeighbors(n_neighbors=5)
neighbors_object.fit(X_train)
distances_of_nearest_neighbors, indices_of_nearest_neighbors_of_query_point = neighbors_object.kneighbors([query_point])
nearest_neighbors_of_query_point = X_train[indices_of_nearest_neighbors_of_query_point[0]]
print("The query point is: {}\n".format(query_point))
print("The nearest neighbors of the query point are:\n {}\n".format(nearest_neighbors_of_query_point))
print("The classes of the nearest neighbors are: {}\n".format(y_train[indices_of_nearest_neighbors_of_query_point[0]]))
print("Predicted class for query point: {}".format(predicted_class_for_query_point[0]))

plt.scatter(X_class0[:, 0], X_class0[:, 1], color='red')
plt.scatter(X_class1[:, 0], X_class1[:, 1], color='blue')
plt.scatter(X_class2[:, 0], X_class2[:, 1], color='green')
plt.scatter(query_point[0], query_point[1], marker='^', s=75, color='black')
plt.scatter(nearest_neighbors_of_query_point[:, 0], nearest_neighbors_of_query_point[:, 1], marker='s', s=150,
            color='yellow', alpha=0.30)
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.legend(['class 0', 'class 1', 'class 2'])
plt.title('Fig 3: Working of the K-NN classification algorithm')
plt.show()

test_set_predictions = [model.predict(X_test[i].reshape((1, len(X_test[i]))))[0] for i in range(X_test.shape[0])]
test_misclassification_percentage = 0
for i in range(len(test_set_predictions)):
    if test_set_predictions[i] != y_test[i]:
        test_misclassification_percentage += 1
test_misclassification_percentage *= 100 / len(y_test)

print("Evaluating K-NN classifier:")
print('test misclassification percentage = {}%'.format(test_misclassification_percentage))
