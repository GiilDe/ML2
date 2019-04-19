from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.naive_bayes import GaussianNB


def test_data_quality(data_X, data_Y, test_X, test_Y):
    clf1 = LinearSVC(random_state=0, tol=1e-5, max_iter=100)
    clf2 = RandomForestClassifier(n_estimators=50, random_state=1)
    clf3 = GaussianNB()
    eclf1 = VotingClassifier(estimators=[('LinearSVC', clf1), ('RandomForestClassifier', clf2), ('GaussianNB', clf3)],
                             voting='hard')
    eclf1.fit(data_X, data_Y)
    prediction = eclf1.predict(test_X)
    print(prediction, test_Y)
    return accuracy_score(test_Y, prediction)
