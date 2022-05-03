"""test_myensembleclassifiers.py
@author lmartin5
"""

from mysklearn.myensembleclassifiers import MyRandomForestClassifier

# note: order is actual/received student value, expected/solution
def test_random_forest_classifier_fit():
    """Test Function
    Note: Due to randomness, just making sure structure is how we want it
    """

    X_train_interview = [
    ["Senior", "Java", "no", "no"],
    ["Senior", "Java", "no", "yes"],
    ["Mid", "Python", "no", "no"],
    ["Junior", "Python", "no", "no"],
    ["Junior", "R", "yes", "no"],
    ["Junior", "R", "yes", "yes"],
    ["Mid", "R", "yes", "yes"],
    ["Senior", "Python", "no", "no"],
    ["Senior", "R", "yes", "no"],
    ["Junior", "Python", "yes", "no"],
    ["Senior", "Python", "yes", "yes"],
    ["Mid", "Python", "no", "yes"],
    ["Mid", "Java", "yes", "no"],
    ["Junior", "Python", "no", "yes"]
    ]
    y_train_interview = ["False", "False", "True", "True", "True", "False", "True", "False", "True", "True", "True", "True", "True", "False"]

    N = 20
    M = 4
    F = 3 # 3 out of 4 attributes selected every time

    forest = MyRandomForestClassifier(N, M, F)
    forest.fit(X_train_interview, y_train_interview)
    trees = forest.trees

    assert len(trees) == M

def test_random_forest_classifier_predict():
    """Test Function
    """

    X_train_interview = [
    ["Senior", "Java", "no", "no"],
    ["Senior", "Java", "no", "yes"],
    ["Mid", "Python", "no", "no"],
    ["Junior", "Python", "no", "no"],
    ["Junior", "R", "yes", "no"],
    ["Junior", "R", "yes", "yes"],
    ["Mid", "R", "yes", "yes"],
    ["Senior", "Python", "no", "no"],
    ["Senior", "R", "yes", "no"],
    ["Junior", "Python", "yes", "no"],
    ["Senior", "Python", "yes", "yes"],
    ["Mid", "Python", "no", "yes"],
    ["Mid", "Java", "yes", "no"],
    ["Junior", "Python", "no", "yes"]
    ]
    y_train_interview = ["False", "False", "True", "True", "True", "False", "True", "False", "True", "True", "True", "True", "True", "False"]

    N = 20
    M = 4
    F = 3 # 3 out of 4 attributes selected every time

    forest = MyRandomForestClassifier(N, M, F)
    forest.fit(X_train_interview, y_train_interview)

    X_test = [["Junior", "Java", "yes", "no"], ["Junior", "Java", "yes", "yes"]]
    y_predicted_solution = ["True", "False"]

    y_predicted = forest.predict(X_test)
    assert len(y_predicted) == len(y_predicted_solution)
    assert y_predicted == y_predicted_solution