# Video tutorial
# https://www.youtube.com/watch?v=84gqSbLcBFE
# How to random split data into training and test sets 
# How to use accuracy_score

from sklearn.datasets import load_iris
iris = load_iris()

X = iris.data
y = iris.target

# Split data
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5)

# Train data
# from sklearn import tree
# my_classifier = tree.DecisionTreeClassifier()
from sklearn.neighbors import KNeighborsClassifier
my_classifier = KNeighborsClassifier()
my_classifier.fit(X_train, y_train)
predictions = my_classifier.predict(X_test)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, predictions))