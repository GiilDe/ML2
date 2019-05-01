from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier


def test_data_quality(data_X, data_Y, test_X, test_Y):
    clf1 = Perceptron(random_state=0, tol=1e-5)
    clf2 = RandomForestClassifier(n_estimators=50, random_state=1)
    clf3 = GaussianNB()
    clf4 = KNeighborsClassifier(n_neighbors=3)
    clf1.fit(data_X, data_Y)
    clf2.fit(data_X, data_Y)
    clf3.fit(data_X, data_Y)
    clf4.fit(data_X, data_Y)
    prediction1 = clf1.predict(test_X)
    prediction2 = clf2.predict(test_X)
    prediction3 = clf3.predict(test_X)
    prediction4 = clf4.predict(test_X)
    accuracy1 = accuracy_score(test_Y, prediction1)
    accuracy2 = accuracy_score(test_Y, prediction2)
    accuracy3 = accuracy_score(test_Y, prediction3)
    accuracy4 = accuracy_score(test_Y, prediction4)

    return accuracy1, accuracy2, accuracy3, accuracy4
