import codecademylib3_seaborn
from sklearn.linear_model import Perceptron
import matplotlib.pyplot as plt
import numpy as np
from itertools import product

#Creating and visualizing AND Data
data = [[0,0], [0,1], [1,0], [1,1]]
labels = [0, 0, 0, 1]

plt.scatter([point[0] for point in data], [point[1] for point in data], c = labels)


#Building the perceptron
classifier = Perceptron(max_iter = 40)
#max_iter sets the number of times the perceptron loops through the training data. The default is 1000, so we’re cutting the training pretty short! Let’s see if our algorithm learns AND even with very little training.
classifier.fit(data, labels)

print(classifier.score(data, labels))
#100%
#Changing data to represen an XOR gate(50% accuracy) and OR gate(100% accuracy)

#Visualizing the Perceptron
print(classifier.decision_function([[0, 0], [1, 1], [0.5, 0.5]]))
#Given a list of points, this method returns the distance those points are from the decision boundary. The closer the number is to 0, the closer that point is to the decision boundary.

#Create a heat map that reveals the decision boundary
x_values = np.linspace(0, 1, 100)
y_values = np.linspace(0, 1, 100)
point_grid = list(product(x_values, y_values))
distances = classifier.decision_function(point_grid)
abs_distances = [abs(pt) for pt in distances]
#Turning abs_distances into a 100x100 list
distances_matrix = np.reshape(abs_distances, (100, 100))
#Plot heat map
heatmap = plt.pcolormesh(x_values, y_values, distances_matrix)
plt.colorbar(heatmap)
plt.show()

#Perceptrons can’t solve problems that aren’t linearly separable(XOR Gate). However, if you combine multiple perceptrons together, you now have a neural net that can solve these problems!

#This is incredibly similar to logic gates. AND gates and OR gates can’t produce the output of XOR gates, but when you combine a few ANDs and ORs, you can make an XOR!