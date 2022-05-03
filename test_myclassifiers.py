"""test_myclassifiers.py
@author lmartin5
"""

from math import sqrt
import numpy as np
from mysklearn.mypytable import MyPyTable

from mysklearn.mysimplelinearregressor import MySimpleLinearRegressor
from mysklearn.myclassifiers import MySimpleLinearRegressionClassifier,\
    MyKNeighborsClassifier,\
    MyDummyClassifier
from mysklearn.myclassifiers import MyNaiveBayesClassifier
from mysklearn.myclassifiers import MyDecisionTreeClassifier
# test_myclassifiers.py solution from PA4-6 below tree tests

# PA4 data
# from in-class #1  (4 instances)
X_train_class_example1 = [[1, 1], [1, 0], [0.33, 0], [0, 0]]
y_train_class_example1 = ["bad", "bad", "good", "good"]

# from in-class #2 (8 instances)
# assume normalized
X_train_class_example2 = [
        [3, 2],
        [6, 6],
        [4, 1],
        [4, 4],
        [1, 2],
        [2, 0],
        [0, 3],
        [1, 6]]
y_train_class_example2 = ["no", "yes", "no", "no", "yes", "no", "yes", "yes"]

# from Bramer
header_bramer_example = ["Attribute 1", "Attribute 2"]
X_train_bramer_example = [
    [0.8, 6.3],
    [1.4, 8.1],
    [2.1, 7.4],
    [2.6, 14.3],
    [6.8, 12.6],
    [8.8, 9.8],
    [9.2, 11.6],
    [10.8, 9.6],
    [11.8, 9.9],
    [12.4, 6.5],
    [12.8, 1.1],
    [14.0, 19.9],
    [14.2, 18.5],
    [15.6, 17.4],
    [15.8, 12.2],
    [16.6, 6.7],
    [17.4, 4.5],
    [18.2, 6.9],
    [19.0, 3.4],
    [19.6, 11.1]]

y_train_bramer_example = ["-", "-", "-", "+", "-", "+", "-", "+", "+", "+", "-", "-", "-",\
           "-", "-", "+", "+", "+", "-", "+"]

# PA6 data
# in-class Naive Bayes example (lab task #1)
inclass_example_col_names = ["att1", "att2"]
X_train_inclass_example = [
    [1, 5], # yes
    [2, 6], # yes
    [1, 5], # no
    [1, 5], # no
    [1, 6], # yes
    [2, 6], # no
    [1, 5], # yes
    [1, 6] # yes
]
y_train_inclass_example = ["yes", "yes", "no", "no", "yes", "no", "yes", "yes"]

# RQ5 (fake) iPhone purchases dataset
iphone_col_names = ["standing", "job_status", "credit_rating", "buys_iphone"]
iphone_table = [
    [1, 3, "fair", "no"],
    [1, 3, "excellent", "no"],
    [2, 3, "fair", "yes"],
    [2, 2, "fair", "yes"],
    [2, 1, "fair", "yes"],
    [2, 1, "excellent", "no"],
    [2, 1, "excellent", "yes"],
    [1, 2, "fair", "no"],
    [1, 1, "fair", "yes"],
    [2, 2, "fair", "yes"],
    [1, 2, "excellent", "yes"],
    [2, 2, "excellent", "yes"],
    [2, 3, "fair", "yes"],
    [2, 2, "excellent", "no"],
    [2, 3, "fair", "yes"]
]

# Bramer 3.2 train dataset
train_col_names = ["day", "season", "wind", "rain", "class"]
train_table = [
    ["weekday", "spring", "none", "none", "on time"],
    ["weekday", "winter", "none", "slight", "on time"],
    ["weekday", "winter", "none", "slight", "on time"],
    ["weekday", "winter", "high", "heavy", "late"],
    ["saturday", "summer", "normal", "none", "on time"],
    ["weekday", "autumn", "normal", "none", "very late"],
    ["holiday", "summer", "high", "slight", "on time"],
    ["sunday", "summer", "normal", "none", "on time"],
    ["weekday", "winter", "high", "heavy", "very late"],
    ["weekday", "summer", "none", "slight", "on time"],
    ["saturday", "spring", "high", "heavy", "cancelled"],
    ["weekday", "summer", "high", "slight", "on time"],
    ["saturday", "winter", "normal", "none", "late"],
    ["weekday", "summer", "high", "none", "on time"],
    ["weekday", "winter", "normal", "heavy", "very late"],
    ["saturday", "autumn", "high", "slight", "on time"],
    ["weekday", "autumn", "none", "heavy", "on time"],
    ["holiday", "spring", "normal", "slight", "on time"],
    ["weekday", "spring", "normal", "none", "on time"],
    ["weekday", "spring", "normal", "slight", "on time"]
]

# PA4 discretizers
def a_discretizer(y_val):
    """Discretizer function used in tests
    """
    if y_val >= 100:
        return "high"
    return "low"

def b_discretizer(y_val):
    """Discretizer function used in tests
    """
    if y_val <= 10:
        return "benchwarmer"
    if y_val <= 20:
        return "rotation player"
    return "starter"

# PA7 Data
# interview dataset
header_interview = ["level", "lang", "tweets", "phd", "interviewed_well"]
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

# bramer degrees dataset
header_degrees = ["SoftEng", "ARIN", "HCI", "CSA", "Project", "Class"]
X_train_degrees = [
    ['A', 'B', 'A', 'B', 'B'],
    ['A', 'B', 'B', 'B', 'A'],
    ['A', 'A', 'A', 'B', 'B'],
    ['B', 'A', 'A', 'B', 'B'],
    ['A', 'A', 'B', 'B', 'A'],
    ['B', 'A', 'A', 'B', 'B'],
    ['A', 'B', 'B', 'B', 'B'],
    ['A', 'B', 'B', 'B', 'B'],
    ['A', 'A', 'A', 'A', 'A'],
    ['B', 'A', 'A', 'B', 'B'],
    ['B', 'A', 'A', 'B', 'B'],
    ['A', 'B', 'B', 'A', 'B'],
    ['B', 'B', 'B', 'B', 'A'],
    ['A', 'A', 'B', 'A', 'B'],
    ['B', 'B', 'B', 'B', 'A'],
    ['A', 'A', 'B', 'B', 'B'],
    ['B', 'B', 'B', 'B', 'B'],
    ['A', 'A', 'B', 'A', 'A'],
    ['B', 'B', 'B', 'A', 'A'],
    ['B', 'B', 'A', 'A', 'B'],
    ['B', 'B', 'B', 'B', 'A'],
    ['B', 'A', 'B', 'A', 'B'],
    ['A', 'B', 'B', 'B', 'A'],
    ['A', 'B', 'A', 'B', 'B'],
    ['B', 'A', 'B', 'B', 'B'],
    ['A', 'B', 'B', 'B', 'B']
]
y_train_degrees = ['SECOND', 'FIRST', 'SECOND', 'SECOND', 'FIRST', 'SECOND',
                   'SECOND', 'SECOND', 'FIRST', 'SECOND', 'SECOND', 'SECOND',
                   'SECOND', 'FIRST', 'SECOND', 'SECOND', 'SECOND', 'FIRST',
                   'SECOND', 'SECOND', 'SECOND', 'SECOND', 'FIRST', 'SECOND',
                   'SECOND', 'SECOND']

# note: order is actual/received student value, expected/solution
def test_decision_tree_classifier_fit():
    """Test function
    """
    # note: this tree uses the generic "att#" attribute labels because fit() does not and should not accept attribute names
    # note: the attribute values are sorted alphabetically
    tree_interview = \
        ["Attribute", "att0",
            ["Value", "Junior",
                ["Attribute", "att3",
                    ["Value", "no",
                        ["Leaf", "True", 3, 5]
                    ],
                    ["Value", "yes",
                        ["Leaf", "False", 2, 5]
                    ]
                ]
            ],
            ["Value", "Mid",
                ["Leaf", "True", 4, 14]
            ],
            ["Value", "Senior",
                ["Attribute", "att2",
                    ["Value", "no",
                        ["Leaf", "False", 3, 5]
                    ],
                    ["Value", "yes",
                        ["Leaf", "True", 2, 5]
                    ]
                ]
            ]
        ]

    tree = MyDecisionTreeClassifier()
    tree.fit(X_train_interview, y_train_interview)
    tree_interview_actual = tree.tree
    assert_correct_tree(tree_interview_actual, tree_interview)

    # note: this tree uses the generic "att#" attribute labels because fit() does not and should not accept attribute names
    # note: the attribute values are sorted alphabetically
    tree_degrees = \
        ["Attribute", "att0",
            ["Value", "A",
                ["Attribute", "att4",
                    ["Value", "A",
                        ["Leaf", "FIRST", 5, 14]
                    ],
                    ["Value", "B",
                        ["Attribute", "att3",
                            ["Value", "A",
                                ["Attribute", "att1",
                                    ["Value", "A",
                                        ["Leaf", "FIRST", 1, 2]
                                    ],
                                    ["Value", "B",
                                        ["Leaf", "SECOND", 1, 2]
                                    ]
                                ]
                            ],
                            ["Value", "B",
                                ["Leaf", "SECOND", 7, 9]
                            ]
                        ]
                    ]
                ]
            ],
            ["Value", "B",
                ["Leaf", "SECOND", 12, 26]
            ],
        ]

    tree = MyDecisionTreeClassifier()
    tree.fit(X_train_degrees, y_train_degrees)
    tree_degrees_actual = tree.tree
    assert_correct_tree(tree_degrees_actual, tree_degrees)

    # note: this tree uses the generic "att#" attribute labels because fit() does not and should not accept attribute names
    # note: the attribute values are sorted alphabetically
    tree_iphone = \
        ["Attribute", "att0",
            ["Value", 1,
                ["Attribute", "att1",
                    ["Value", 1,
                        ["Leaf", "yes", 1, 5]
                    ],
                    ["Value", 2,
                        ["Attribute", "att2",
                            ["Value", "excellent",
                                ["Leaf", "yes", 1, 2]
                            ],
                            ["Value", "fair",
                                ["Leaf", "no", 1, 2]
                            ]
                        ]
                    ],
                    ["Value", 3,
                        ["Leaf", "no", 2, 5]
                    ]
                ]
            ],
            ["Value", 2,
                ["Attribute", "att2",
                    ["Value", "excellent",
                        ["Leaf", "no", 4, 10]
                    ],
                    ["Value", "fair",
                        ["Leaf", "yes", 6, 10]
                    ]
                ]
            ]
        ]

    # serparating attributes and class
    iphone_data = MyPyTable(iphone_col_names, iphone_table)
    y_train = iphone_data.get_column("buys_iphone")
    iphone_data.drop_column("buys_iphone")
    X_train = iphone_data.data

    tree = MyDecisionTreeClassifier()
    tree.fit(X_train, y_train)
    tree_iphone_actual = tree.tree
    assert_correct_tree(tree_iphone_actual, tree_iphone)

def assert_correct_tree(actual_tree, solution_tree):
    """Test function
    Recursively defined to check tree is the same
    """
    assert len(actual_tree) == len(solution_tree)
    assert actual_tree[0] == solution_tree[0]

    # base case for test
    if actual_tree[0] == "Leaf":
        # will just be a 1D list in this case
        # all discrete values for trees
        assert actual_tree == solution_tree
        return

    if actual_tree[0] == "Attribute":
        # the attribute name (i.e. 'att0')
        assert actual_tree[1] == solution_tree[1]
        # looping through all value subtrees
        for actual_subtree, solution_subtree in zip(actual_tree[2:], solution_tree[2:]):
            assert_correct_tree(actual_subtree, solution_subtree)

    if actual_tree[0] == "Value":
        # the attrib. value (i.e. "yes")
        assert actual_tree[1] == solution_tree[1]
        # checking the subtree
        assert_correct_tree(actual_tree[2], solution_tree[2])

def test_decision_tree_classifier_predict():
    """Test function
    """
    tree_interview = \
        ["Attribute", "att0",
            ["Value", "Junior",
                ["Attribute", "att3",
                    ["Value", "no",
                        ["Leaf", "True", 3, 5]
                    ],
                    ["Value", "yes",
                        ["Leaf", "False", 2, 5]
                    ]
                ]
            ],
            ["Value", "Mid",
                ["Leaf", "True", 4, 14]
            ],
            ["Value", "Senior",
                ["Attribute", "att2",
                    ["Value", "no",
                        ["Leaf", "False", 3, 5]
                    ],
                    ["Value", "yes",
                        ["Leaf", "True", 2, 5]
                    ]
                ]
            ]
        ]

    X_test = [["Junior", "Java", "yes", "no"], ["Junior", "Java", "yes", "yes"]]
    y_predicted_solution = ["True", "False"]
    tree = MyDecisionTreeClassifier()
    tree.tree = tree_interview
    y_predicted = tree.predict(X_test)
    assert len(y_predicted) == len(y_predicted_solution)
    assert y_predicted == y_predicted_solution

    tree_degrees = \
        ["Attribute", "att0",
            ["Value", "A",
                ["Attribute", "att4",
                    ["Value", "A",
                        ["Leaf", "FIRST", 5, 14]
                    ],
                    ["Value", "B",
                        ["Attribute", "att3",
                            ["Value", "A",
                                ["Attribute", "att1",
                                    ["Value", "A",
                                        ["Leaf", "FIRST", 1, 2]
                                    ],
                                    ["Value", "B",
                                        ["Leaf", "SECOND", 1, 2]
                                    ]
                                ]
                            ],
                            ["Value", "B",
                                ["Leaf", "SECOND", 7, 9]
                            ]
                        ]
                    ]
                ]
            ],
            ["Value", "B",
                ["Leaf", "SECOND", 12, 26]
            ],
        ]

    X_test = [["B", "B", "B", "B", "B"], ["A", "A", "A", "A", "A"], ["A", "A", "A", "A", "B"]]
    y_predicted_solution = ["SECOND", "FIRST", "FIRST"]
    tree = MyDecisionTreeClassifier()
    tree.tree = tree_degrees
    y_predicted = tree.predict(X_test)
    assert len(y_predicted) == len(y_predicted_solution)
    assert y_predicted == y_predicted_solution

    tree_iphone = \
        ["Attribute", "att0",
            ["Value", 1,
                ["Attribute", "att1",
                    ["Value", 1,
                        ["Leaf", "yes", 1, 5]
                    ],
                    ["Value", 2,
                        ["Attribute", "att2",
                            ["Value", "excellent",
                                ["Leaf", "yes", 1, 2]
                            ],
                            ["Value", "fair",
                                ["Leaf", "no", 1, 2]
                            ]
                        ]
                    ],
                    ["Value", 3,
                        ["Leaf", "no", 2, 5]
                    ]
                ]
            ],
            ["Value", 2,
                ["Attribute", "att2",
                    ["Value", "excellent",
                        ["Leaf", "no", 4, 10]
                    ],
                    ["Value", "fair",
                        ["Leaf", "yes", 6, 10]
                    ]
                ]
            ]
        ]

    X_test = [[2, 2, "fair"], [1, 1, "excellent"]]
    y_predicted_solution = ["yes", "yes"]
    tree = MyDecisionTreeClassifier()
    tree.tree = tree_iphone
    y_predicted = tree.predict(X_test)
    assert len(y_predicted) == len(y_predicted_solution)
    assert y_predicted == y_predicted_solution

def test_simple_linear_regression_classifier_fit():
    """Test function
    """
    np.random.seed(0)

    # test a
    X_train = [[val] for val in range(100)] # 2D list, but only 1 column
    y_train = [row[0] * 2 + np.random.normal(0, 25) for row in X_train] # data with noise
    a_lin_reg_classifier = MySimpleLinearRegressionClassifier(a_discretizer)
    a_lin_reg_classifier.fit(X_train, y_train)

    slope_solution = 1.924917458430444
    intercept_solution = 5.211786196055144
    assert np.isclose(a_lin_reg_classifier.regressor.slope, slope_solution)
    assert np.isclose(a_lin_reg_classifier.regressor.intercept, intercept_solution)

    # test b (fake player points/minutes data)
    X_train = [[34.5], [13.4], [15.5], [17.8], [38.3], [22.2], [16.8], [13.6]]
    y_train = [40.0, 12.6, 13.8, 24.5, 30.1, 18.0, 14.5, 12.6]
    b_lin_reg_classifier = MySimpleLinearRegressionClassifier(b_discretizer)
    b_lin_reg_classifier.fit(X_train, y_train)

    # caclulated
    slope_solution = 0.9101728051
    intercept_solution = 1.18240753
    assert np.isclose(b_lin_reg_classifier.regressor.slope, slope_solution)
    assert np.isclose(b_lin_reg_classifier.regressor.intercept, intercept_solution)

def test_simple_linear_regression_classifier_predict():
    """Test function
    """
    # test a
    m = 1.924917458430444
    b = 5.211786196055144
    lin_reg = MySimpleLinearRegressor(m, b)

    a_lin_reg_classifier = MySimpleLinearRegressionClassifier(a_discretizer, lin_reg)

    # hand calculated predictions using slope, intercept numbers from above test
    X_test = [[35.6], [18.7], [102.5], [0], [49], [76.3], [50.1], [78.3], [6]]
    y_predicted_actual = ["low", "low", "high", "low", "low", "high", "high", "high", "low"]
    y_predicted = a_lin_reg_classifier.predict(X_test)
    assert y_predicted == y_predicted_actual

    # test b
    m = 0.9101728051
    b = 1.18240753
    lin_reg = MySimpleLinearRegressor(m, b)
    b_lin_reg_classifier = MySimpleLinearRegressionClassifier(b_discretizer, lin_reg)

    # predictions through sklearn
    X_test = [[24.5], [0], [6.5], [15.8], [39.4], [26.7], [12.3], [24.0], [16]]
    y_predicted_actual = ["starter", "benchwarmer", "benchwarmer", "rotation player",
                          "starter", "starter", "rotation player", "starter", "rotation player"]
    y_predicted = b_lin_reg_classifier.predict(X_test)
    assert y_predicted == y_predicted_actual

def test_kneighbors_classifier_kneighbors():
    """Test function
    """
    # test a
    knn = MyKNeighborsClassifier(3)
    knn.fit(X_train_class_example1, y_train_class_example1)

    X_test = [[1/3, 1]]
    distances, neighbors = knn.kneighbors(X_test)
    distances_actual = [2/3, 1, sqrt(10/9)]
    neighbors_actual = [0, 2, 3]
    assert len(distances) == 1
    assert len(neighbors) == 1
    assert np.allclose(distances[0], distances_actual)
    assert neighbors[0] == neighbors_actual

    # test b
    knn = MyKNeighborsClassifier(3)
    knn.fit(X_train_class_example2, y_train_class_example2)

    X_test = [[2, 3]]
    distances, neighbors = knn.kneighbors(X_test)
    distances_actual = [1.4142135623730951, 1.4142135623730951, 2.0]
    neighbors_actual = [0, 4, 6]
    assert len(distances) == 1
    assert len(neighbors) == 1
    assert np.allclose(distances[0], distances_actual)
    assert neighbors[0] == neighbors_actual

    # test c
    knn = MyKNeighborsClassifier(5)
    knn.fit(X_train_bramer_example, y_train_bramer_example)

    X_test = [[9.1, 11.0]]
    distances, neighbors = knn.kneighbors(X_test)
    distances_actual = [0.608, 1.237, 2.202, 2.802, 2.915]
    neighbors_actual = [6, 5, 7, 4, 8]
    assert len(distances) == 1
    assert len(neighbors) == 1
    # makes it so distances only have to be within 10^-3 accurate to neighbors_actual
    assert np.allclose(distances[0], distances_actual, atol=10**(-3))
    assert neighbors[0] == neighbors_actual

def test_kneighbors_classifier_predict():
    """Test function
    """
    # test a
    knn = MyKNeighborsClassifier(3)
    knn.fit(X_train_class_example1, y_train_class_example1)

    X_test = [[1/3, 1]]
    y_predicted = knn.predict(X_test)
    y_predicted_actual = ["good"]
    assert len(y_predicted) == 1
    assert y_predicted[0] == y_predicted_actual[0]
    assert y_predicted == y_predicted_actual

    # test b
    knn = MyKNeighborsClassifier(3)
    knn.fit(X_train_class_example2, y_train_class_example2)

    X_test = [[2, 3]]
    y_predicted = knn.predict(X_test)
    y_predicted_actual = ["yes"]
    assert len(y_predicted) == 1
    assert y_predicted[0] == y_predicted_actual[0]
    assert y_predicted == y_predicted_actual

    # test c
    knn = MyKNeighborsClassifier(5)
    knn.fit(X_train_bramer_example, y_train_bramer_example)

    X_test = [[9.1, 11.0]]
    y_predicted = knn.predict(X_test)
    y_predicted_actual = ["+"]
    assert len(y_predicted) == 1
    assert y_predicted[0] == y_predicted_actual[0]
    assert y_predicted == y_predicted_actual

def test_dummy_classifier_fit():
    """Test function
    """
    np.random.seed(0)
    # test a
    X_train = list(range(100))
    y_train = list(np.random.choice(["yes", "no"], 100, replace=True, p=[0.7, 0.3]))
    dummy = MyDummyClassifier()
    dummy.fit(X_train, y_train)
    assert dummy.most_common_label == "yes"

    # test b
    X_train = list(range(100))
    y_train = list(np.random.choice(["yes", "no", "maybe"], 100, replace=True, p=[0.2, 0.6, 0.2]))
    dummy = MyDummyClassifier()
    dummy.fit(X_train, y_train)
    assert dummy.most_common_label == "no"

    # test c
    X_train = [["Game " + str(i)] for i in range(1, 83)]
    y_train = list(np.random.choice(["Ben Simmons played", "Ben Simmons did not play"], 82, replace=True, p=[0.01, 0.99]))
    dummy = MyDummyClassifier()
    dummy.fit(X_train, y_train)
    assert dummy.most_common_label == "Ben Simmons did not play"

def test_dummy_classifier_predict():
    """Test function
    """
    np.random.seed(0)
    # test a
    X_train = list(range(100))
    y_train = list(np.random.choice(["yes", "no"], 100, replace=True, p=[0.7, 0.3]))
    dummy = MyDummyClassifier()
    dummy.fit(X_train, y_train)

    X_test = [1009, 96, -365]
    y_predicted = dummy.predict(X_test)
    y_predicted_actual = ["yes", "yes", "yes"]
    assert len(y_predicted) == 3
    assert y_predicted == y_predicted_actual

    # test b
    X_train = list(range(100))
    y_train = list(np.random.choice(["yes", "no", "maybe"], 100, replace=True, p=[0.2, 0.6, 0.2]))
    dummy = MyDummyClassifier()
    dummy.fit(X_train, y_train)

    X_test = ["test", "case", "here"]
    y_predicted = dummy.predict(X_test)
    y_predicted_actual = ["no", "no", "no"]
    assert len(y_predicted) == 3
    assert y_predicted == y_predicted_actual

    # test c
    X_train = list(range(100))
    y_train = list(np.random.choice(["Ben Simmons played", "Ben Simmons did not play"], 100, replace=True, p=[0.01, 0.99]))
    dummy = MyDummyClassifier()
    dummy.fit(X_train, y_train)

    X_test = ["Game 83"]
    y_predicted = dummy.predict(X_test)
    y_predicted_actual = ["Ben Simmons did not play"]
    assert len(y_predicted) == 1
    assert y_predicted == y_predicted_actual

def check_correct_posteriors(posteriors_actual, posteriors_solution):
    """Test function
    Function tests the posteriors regardless of order (for keys) except for attribute order,
    which should be consistent if fit is implemented regularly
    """
    assert len(posteriors_actual.keys()) == len(posteriors_solution.keys())
    for key in posteriors_solution.keys():
        assert len(posteriors_actual[key]) == len(posteriors_solution[key])
        for prob_actual, prob_solution in zip(posteriors_actual[key], posteriors_solution[key]):
            assert len(prob_actual.keys()) == len(prob_solution.keys())
            for key2 in prob_solution.keys():
                assert np.isclose(prob_actual[key2], prob_solution[key2])

def test_naive_bayes_classifier_fit():
    """Test function
    """
    # test a - ipad example test
    X = X_train_inclass_example
    y = y_train_inclass_example

    nb_classifier = MyNaiveBayesClassifier()
    nb_classifier.fit(X, y)
    priors_actual = nb_classifier.priors
    posteriors_actual = nb_classifier.posteriors

    priors_solution = {"no": 3/8, "yes": 5/8}
    posteriors_solution = {"no": [{1: 2/3, 2: 1/3}, {5: 2/3, 6: 1/3}],
                            "yes": [{1: 4/5, 2: 1/5}, {5: 2/5, 6: 3/5}]}

    assert len(priors_actual.keys()) == len(priors_solution.keys())
    for key, probability in priors_solution.items():
        assert np.isclose(priors_actual[key], probability)
    check_correct_posteriors(posteriors_actual, posteriors_solution)

    # test b - RQ5 test
    iphone_data = MyPyTable(iphone_col_names, iphone_table)
    y = iphone_data.get_column("buys_iphone")
    iphone_data.drop_column("buys_iphone")
    X = iphone_data.data

    nb_classifier = MyNaiveBayesClassifier()
    nb_classifier.fit(X, y)
    priors_actual = nb_classifier.priors
    posteriors_actual = nb_classifier.posteriors

    priors_solution = {"no": 5/15, "yes": 10/15}
    posteriors_solution = {"no": [{1: 3/5, 2: 2/5}, {1: 1/5, 2: 2/5, 3: 2/5}, {"excellent": 3/5, "fair": 2/5}],
                            "yes": [{1: 2/10, 2: 8/10}, {1: 3/10, 2: 4/10, 3: 3/10}, {"excellent": 3/10, "fair": 7/10}]}

    assert len(priors_actual.keys()) == len(priors_solution.keys())
    for key, probability in priors_solution.items():
        assert np.isclose(priors_actual[key], probability)
    check_correct_posteriors(posteriors_actual, posteriors_solution)

    # test c - bramer test
    bramer_train_data = MyPyTable(train_col_names, train_table)
    y = bramer_train_data.get_column("class")
    bramer_train_data.drop_column("class")
    X = bramer_train_data.data

    nb_classifier = MyNaiveBayesClassifier()
    nb_classifier.fit(X, y)
    priors_actual = nb_classifier.priors
    posteriors_actual = nb_classifier.posteriors

    priors_solution = {"on time": 14/20, "late": 2/20, "very late": 3/20, "cancelled": 1/20}
    posteriors_solution = {"on time": [{"weekday": 9/14, "saturday": 2/14, "sunday": 1/14, "holiday": 2/14},
                                       {"spring": 4/14, "summer": 6/14, "autumn": 2/14, "winter": 2/14},
                                       {"none": 5/14, "high": 4/14, "normal": 5/14},
                                       {"none": 5/14, "slight": 8/14, "heavy": 1/14}],
                           "late": [{"weekday": 1/2, "saturday": 1/2, "sunday": 0/2, "holiday": 0/2},
                                    {"spring": 0/2, "summer": 0/2, "autumn": 0/2, "winter": 2/2},
                                    {"none": 0/2, "high": 1/2, "normal": 1/2},
                                    {"none": 1/2, "slight": 0/2, "heavy": 1/2}],
                           "very late": [{"weekday": 3/3, "saturday": 0/3, "sunday": 0/3, "holiday": 0/3},
                                         {"spring": 0/3, "summer": 0/3, "autumn": 1/3, "winter": 2/3},
                                         {"none": 0/3, "high": 1/3, "normal": 2/3},
                                         {"none": 1/3, "slight": 0/3, "heavy": 2/3}],
                           "cancelled": [{"weekday": 0/1, "saturday": 1/1, "sunday": 0/1, "holiday": 0/1},
                                        {"spring": 1/1, "summer": 0/1, "autumn": 0/1, "winter": 0/1},
                                        {"none": 0/1, "high": 1/1, "normal": 0/1},
                                        {"none": 0/1, "slight": 0/1, "heavy": 1/1}]}

    assert len(priors_actual.keys()) == len(priors_solution.keys())
    for key, probability in priors_solution.items():
        assert np.isclose(priors_actual[key], probability)
    check_correct_posteriors(posteriors_actual, posteriors_solution)

def test_naive_bayes_classifier_predict():
    """Test function
    """
    # test a - ipad example test
    X_test = [[1, 5]]

    priors = {"no": 3/8, "yes": 5/8}
    posteriors = {"no": [{1: 2/3, 2: 1/3}, {5: 2/3, 6: 1/3}],
                            "yes": [{1: 4/5, 2: 1/5}, {5: 2/5, 6: 3/5}]}
    nb_classifier = MyNaiveBayesClassifier(priors, posteriors)
    y_predicted = nb_classifier.predict(X_test)

    y_pred_solution = ["yes"]
    assert len(y_predicted) == len(y_pred_solution)
    assert y_predicted == y_pred_solution

    # test b - RQ5 test
    X_test = [[2, 2, "fair"], [1, 1, "excellent"]]

    priors = {"no": 5/15, "yes": 10/15}
    posteriors = {"no": [{1: 3/5, 2: 2/5}, {1: 1/5, 2: 2/5, 3: 2/5}, {"excellent": 3/5, "fair": 2/5}],
                            "yes": [{1: 2/10, 2: 8/10}, {1: 3/10, 2: 4/10, 3: 3/10}, {"excellent": 3/10, "fair": 7/10}]}
    nb_classifier = MyNaiveBayesClassifier(priors, posteriors)
    y_predicted = nb_classifier.predict(X_test)

    y_pred_solution = ["yes", "no"]
    assert len(y_predicted) == len(y_pred_solution)
    assert y_predicted == y_pred_solution

    # test c - bramer test
    X_test = [["weekday", "winter", "high", "heavy"], ["weekday", "summer", "high", "heavy"], ["sunday", "summer", "normal", "slight"]]

    priors = {"on time": 14/20, "late": 2/20, "very late": 3/20, "cancelled": 1/20}
    posteriors = {"on time": [{"weekday": 9/14, "saturday": 2/14, "sunday": 1/14, "holiday": 2/14},
                              {"spring": 4/14, "summer": 6/14, "autumn": 2/14, "winter": 2/14},
                              {"none": 5/14, "high": 4/14, "normal": 5/14},
                              {"none": 5/14, "slight": 8/14, "heavy": 1/14}],
                  "late": [{"weekday": 1/2, "saturday": 1/2, "sunday": 0/2, "holiday": 0/2},
                           {"spring": 0/2, "summer": 0/2, "autumn": 0/2, "winter": 2/2},
                           {"none": 0/2, "high": 1/2, "normal": 1/2},
                           {"none": 1/2, "slight": 0/2, "heavy": 1/2}],
                  "very late": [{"weekday": 3/3, "saturday": 0/3, "sunday": 0/3, "holiday": 0/3},
                                {"spring": 0/3, "summer": 0/3, "autumn": 1/3, "winter": 2/3},
                                {"none": 0/3, "high": 1/3, "normal": 2/3},
                                {"none": 1/3, "slight": 0/3, "heavy": 2/3}],
                  "cancelled": [{"weekday": 0/1, "saturday": 1/1, "sunday": 0/1, "holiday": 0/1},
                                {"spring": 1/1, "summer": 0/1, "autumn": 0/1, "winter": 0/1},
                                {"none": 0/1, "high": 1/1, "normal": 0/1},
                                {"none": 0/1, "slight": 0/1, "heavy": 1/1}]}
    nb_classifier = MyNaiveBayesClassifier(priors, posteriors)
    y_predicted = nb_classifier.predict(X_test)

    y_pred_solution = ["very late", "on time", "on time"]
    assert len(y_predicted) == len(y_pred_solution)
    assert y_predicted == y_pred_solution
