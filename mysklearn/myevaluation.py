"""PA7
Name: Luke Martin
Clss: CPSC 322 - 02
Sems: Spring 2022
Date: April 12, 2022

mysklearn.myevaluation.py
Description:
    This python file contains the functions for evaluating classifiers.
    The functions are:
        train_test_split()
        kfold_cross_validation()
        stratified_kfold_cross_validation()
        bootstrap_sample()
        confusion_matrix()
        accuracy_score()
        binary_precision_score()
        binary_recall_score()
        binary_f1_score()
"""

from copy import deepcopy
from numpy import random # used only for picking random indexes for selection
from tabulate import tabulate # used for confusion matrices

from mysklearn import myutils
from mysklearn.mypytable import MyPyTable

def train_test_split(X, y, test_size=0.33, random_state=None, shuffle=True):
    """Split dataset into train and test sets based on a test set size.

    Args:
        X(list of list of obj): The list of samples
            The shape of X is (n_samples, n_features)
        y(list of obj): The target y values (parallel to X)
            The shape of y is n_samples
        test_size(float or int): float for proportion of dataset to be in test set (e.g. 0.33 for a 2:1 split)
            or int for absolute number of instances to be in test set (e.g. 5 for 5 instances in test set)
        random_state(int): integer used for seeding a random number generator for reproducible results
            Use random_state to seed your random number generator
                you can use the math module or use numpy for your generator
                choose one and consistently use that generator throughout your code
        shuffle(bool): whether or not to randomize the order of the instances before splitting
            Shuffle the rows in X and y before splitting and be sure to maintain the parallel order of X and y!!

    Returns:
        X_train(list of list of obj): The list of training samples
        X_test(list of list of obj): The list of testing samples
        y_train(list of obj): The list of target y values for training (parallel to X_train)
        y_test(list of obj): The list of target y values for testing (parallel to X_test)

    Note:
        Loosely based on sklearn's train_test_split():
            https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
    """
    if isinstance(test_size, float):
        train_size = int(len(X) * (1 - test_size))
    elif isinstance(test_size, int):
        train_size = len(X) - test_size
    else:
        raise TypeError("test_size needs to be either a float or an int")

    if random_state is None:
        random.seed()
    else:
        random.seed(random_state)

    indices = list(range(len(X)))
    if shuffle:
        for i in range(len(X)):
            rand_index = int(random.randint(0, len(indices)))
            indices[i], indices[rand_index] = indices[rand_index], indices[i]

    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

    X_train = [X[i] for i in train_indices]
    y_train = [y[i] for i in train_indices]
    X_test = [X[i] for i in test_indices]
    y_test = [y[i] for i in test_indices]

    return X_train, X_test, y_train, y_test

def kfold_cross_validation(X, n_splits=5, random_state=None, shuffle=False):
    """Split dataset into cross validation folds.

    Args:
        X(list of list of obj): The list of samples
            The shape of X is (n_samples, n_features)
        n_splits(int): Number of folds.
        random_state(int): integer used for seeding a random number generator for reproducible results
        shuffle(bool): whether or not to randomize the order of the instances before creating folds
    Returns:
        X_train_folds(list of list of int): The list of training set indices for each fold
        X_test_folds(list of list of int): The list of testing set indices for each fold

    Notes:
        The first n_samples % n_splits folds have size n_samples // n_splits + 1,
            other folds have size n_samples // n_splits, where n_samples is the number of samples
            (e.g. 11 samples and 4 splits, the sizes of the 4 folds are 3, 3, 3, 2 samples)
        Loosely based on sklearn's KFold split():
            https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html
    """
    if random_state is None:
        random.seed()
    else:
        random.seed(random_state)

    indices = list(range(len(X)))
    if shuffle:
        for i in range(len(X)):
            rand_index = int(random.randint(0, len(indices)))
            indices[i], indices[rand_index] = indices[rand_index], indices[i]

    sets = [[] for _ in range(n_splits)]
    curr_set = 0
    for index in indices:
        sets[curr_set].append(index)
        curr_set = (curr_set + 1) % n_splits

    X_train_folds = [[] for _ in range(n_splits)]
    X_test_folds = [[] for _ in range(n_splits)]
    for i in range(n_splits):
        test_set = deepcopy(sets[i])

        train_set = []
        for j in range(n_splits):
            if j != i:
                train_set += deepcopy(sets[j])

        X_test_folds[i] = test_set
        X_train_folds[i] = train_set

    return X_train_folds, X_test_folds

def stratified_kfold_cross_validation(X, y, n_splits=5, random_state=None, shuffle=False):
    """Split dataset into stratified cross validation folds.

    Args:
        X(list of list of obj): The list of instances (samples).
            The shape of X is (n_samples, n_features)
        y(list of obj): The target y values (parallel to X).
            The shape of y is n_samples
        n_splits(int): Number of folds.
        random_state(int): integer used for seeding a random number generator for reproducible results
        shuffle(bool): whether or not to randomize the order of the instances before creating folds
    Returns:
        X_train_folds(list of list of int): The list of training set indices for each fold.
        X_test_folds(list of list of int): The list of testing set indices for each fold.

    Notes:
        Loosely based on sklearn's StratifiedKFold split():
            https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html#sklearn.model_selection.StratifiedKFold
    """
    if random_state is None:
        random.seed()
    else:
        random.seed(random_state)

    indices = list(range(len(X)))
    if shuffle:
        for i in range(len(X)):
            rand_index = int(random.randint(0, len(indices)))
            indices[i], indices[rand_index] = indices[rand_index], indices[i]

    class_indices = myutils.get_indicies_per_class(y, indices)

    sets = [[] for _ in range(n_splits)]
    curr_set = 0
    for _, indices in class_indices.items():
        for index in indices:
            sets[curr_set].append(index)
            curr_set = (curr_set + 1) % n_splits

    X_train_folds = [[] for _ in range(n_splits)]
    X_test_folds = [[] for _ in range(n_splits)]
    for i in range(n_splits):
        test_set = deepcopy(sets[i])
        train_set = []
        for j in range(n_splits):
            if j != i:
                train_set += deepcopy(sets[j])

        X_test_folds[i] = test_set
        X_train_folds[i] = train_set

    return X_train_folds, X_test_folds

def bootstrap_sample(X, y=None, n_samples=None, random_state=None):
    """Split dataset into bootstrapped training set and out of bag test set.

    Args:
        X(list of list of obj): The list of samples
        y(list of obj): The target y values (parallel to X)
            Default is None (in this case, the calling code only wants to sample X)
        n_samples(int): Number of samples to generate. If left to None (default) this is automatically
            set to the first dimension of X.
        random_state(int): integer used for seeding a random number generator for reproducible results
    Returns:
        X_sample(list of list of obj): The list of samples
        X_out_of_bag(list of list of obj): The list of "out of bag" samples (e.g. left-over samples)
        y_sample(list of obj): The list of target y values sampled (parallel to X_sample)
            None if y is None
        y_out_of_bag(list of obj): The list of target y values "out of bag" (parallel to X_out_of_bag)
            None if y is None
    Notes:
        Loosely based on sklearn's resample():
            https://scikit-learn.org/stable/modules/generated/sklearn.utils.resample.html
    """
    if y is None:
        y_vals = False
        y_sample = None
        y_out_of_bag = None
    else:
        y_vals = True

    if n_samples is None:
        n_samples = len(X)

    if random_state is None:
        random.seed()
    else:
        random.seed(random_state)

    sampled_indexes = []
    for i in range(n_samples):
        sampled_indexes.append(random.randint(0, len(X)))
    X_sample = [deepcopy(X[i]) for i in sampled_indexes]
    X_out_of_bag = [deepcopy(X[i]) for i in range(len(X)) if i not in sampled_indexes]
    if y_vals:
        y_sample = [deepcopy(y[i]) for i in sampled_indexes]
        y_out_of_bag = [deepcopy(y[i]) for i in range(len(X)) if i not in sampled_indexes]

    return X_sample, X_out_of_bag, y_sample, y_out_of_bag

def confusion_matrix(y_true, y_pred, labels):
    """Compute confusion matrix to evaluate the accuracy of a classification.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of str): The list of all possible target y labels used to index the matrix

    Returns:
        matrix(list of list of int): Confusion matrix whose i-th row and j-th column entry
            indicates the number of samples with true label being i-th class
            and predicted label being j-th class

    Notes:
        Loosely based on sklearn's confusion_matrix():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
    """
    matrix = [[0] * len(labels) for _ in range(len(labels))]
    for i, _ in enumerate(y_true):
        actual_index = labels.index(y_true[i])
        predicted_index = labels.index(y_pred[i])
        matrix[actual_index][predicted_index] += 1

    return matrix

def accuracy_score(y_true, y_pred, normalize=True):
    """Compute the classification prediction accuracy score.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        normalize(bool): If False, return the number of correctly classified samples.
            Otherwise, return the fraction of correctly classified samples.

    Returns:
        score(float): If normalize == True, return the fraction of correctly classified samples (float),
            else returns the number of correctly classified samples (int).

    Notes:
        Loosely based on sklearn's accuracy_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html#sklearn.metrics.accuracy_score
    """
    num_correct = sum([1 for i in range(len(y_pred)) if y_pred[i] == y_true[i]])
    num_total = len(y_pred)

    if normalize:
        score = num_correct / num_total
    else:
        score = num_correct
    return score

def sub_sampling_accuracy(X, y, classifier, k=5, test_size=0.33, random_state=None, shuffle=True, normalize=True):
    """Split dataset into train and test sets based on a test set size k times and compute average accuracy.

    Args:
        X(list of list of obj): The list of samples
            The shape of X is (n_samples, n_features)
        y(list of obj): The target y values (parallel to X)
            The shape of y is n_samples
        classifier(obj): one of the classifiers from mysklearn to evaluate
        k(int): number of times to evaluate the classifier using train_test_split()
        test_size(float or int): float for proportion of dataset to be in test set (e.g. 0.33 for a 2:1 split)
            or int for absolute number of instances to be in test set (e.g. 5 for 5 instances in test set)
        random_state(int): integer used for seeding a random number generator for reproducible results
        shuffle(bool): whether or not to randomize the order of the instances before splitting
        normalize(bool): whether or not to normalize the accuracy of the classifier

    Returns:
        accuracy(float): the average accuracy of the classifier over k train_test_split() calls
    """
    correct = 0
    total_predictions = 0
    for _ in range(k):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size, random_state, shuffle)

        classifier.fit(X_train, y_train)
        y_predicted = classifier.predict(X_test)
        correct += accuracy_score(y_test, y_predicted, normalize=False)
        total_predictions += len(y_predicted)


    if normalize:
        accuracy = correct / total_predictions
    else:
        accuracy = correct

    return accuracy

def kfold_accuracy(X, y, classifier, n_splits=5, stratified=True, random_state=None, shuffle=True, normalize=True):
    """Split dataset into train and test folds using the kfold_cross_validation functions and return accuracy of predictions.
    Accuracy is computed with the average accuracy over n_split runs, using a different fold as the test
    set every time.

    Args:
        X(list of list of obj): The list of samples
            The shape of X is (n_samples, n_features)
        y(list of obj): The target y values (parallel to X)
            The shape of y is n_samples
        classifier(obj): one of the classifiers from mysklearn to evaluate
        n_splits(int): Number of folds
        stratified(bool): whether to use to the stratified_kfold function or the regular kfold function
        random_state(int): integer used for seeding a random number generator for reproducible results
        shuffle(bool): whether or not to randomize the order of the instances before splitting
        normalize(bool): whether or not to normalize the accuracy of the classifier

    Returns:
        accuracy(float): the average accuracy of the classifier over n_split train, test runs
    """
    if stratified:
        X_train_folds, X_test_folds = stratified_kfold_cross_validation(X, y, n_splits, random_state, shuffle)
    else:
        X_train_folds, X_test_folds = kfold_cross_validation(X, n_splits, random_state, shuffle)

    correct = 0
    total_predictions = 0
    for i in range(n_splits):
        X_train = [X[j] for j in X_train_folds[i]]
        X_test = [X[j] for j in X_test_folds[i]]
        y_train = [y[j] for j in X_train_folds[i]]
        y_test = [y[j] for j in X_test_folds[i]]

        classifier.fit(X_train, y_train)
        y_predicted = classifier.predict(X_test)
        correct += accuracy_score(y_test, y_predicted, normalize=False)
        total_predictions += len(y_predicted)

    if normalize:
        accuracy = correct / total_predictions
    else:
        accuracy = correct

    return accuracy

def bootstrap_sampling_accuracy(X, y, classifier, k=5, n_samples=None, random_state=None, normalize=True):
    """Split dataset into train and test sets using the bootstrap method k times and compute average accuracy.

    Args:
        X(list of list of obj): The list of samples
            The shape of X is (n_samples, n_features)
        y(list of obj): The target y values (parallel to X)
            The shape of y is n_samples
        classifier(obj): one of the classifiers from mysklearn to evaluate
        k(int): number of times to evaluate the classifier using train_test_split()
        n_samples(int): Number of samples to generate. If left to None (default) this is automatically
            set to the first dimension of X.
        random_state(int): integer used for seeding a random number generator for reproducible results
        normalize(bool): whether or not to normalize the accuracy of the classifier

    Returns:
        accuracy(float): the average accuracy of the classifier over k runs using bootstrap_sampling
    """
    correct = 0
    total_predictions = 0
    for _ in range(k):
        X_train, X_test, y_train, y_test = bootstrap_sample(X, y, n_samples, random_state)

        classifier.fit(X_train, y_train)
        y_predicted = classifier.predict(X_test)
        correct += accuracy_score(y_test, y_predicted, normalize=False)
        total_predictions += len(y_predicted)

    if normalize:
        accuracy = correct / total_predictions
    else:
        accuracy = correct

    return accuracy

def kfold_confusion_matrix(X, y, labels, classifier, n_splits=5, stratified=True, random_state=None, shuffle=True, label=""):
    """Store the results of using the kfold_cross_validation splitting and predicting in a confusion matrix.
    Additional labels are added onto the matrix (total, recogntion %).

    Args:
        X(list of list of obj): The list of samples
            The shape of X is (n_samples, n_features)
        y(list of obj): The target y values (parallel to X)
            The shape of y is n_samples
        labels(list of obj): The unique values of the label (espicially in case one is never used in sample)
        classifier(obj): one of the classifiers from mysklearn to evaluate
        n_splits(int): Number of folds
        stratified(bool): whether to use to the stratified_kfold function or the regular kfold function
        random_state(int): integer used for seeding a random number generator for reproducible results
        shuffle(bool): whether or not to randomize the order of the instances before splitting
        label(str): the string label to put in the top left corner of the matrix (the class attribute name)

    Returns:
        accuracy(float): the average accuracy of the classifier over n_split train, test runs
    """

    if stratified:
        X_train_folds, X_test_folds = stratified_kfold_cross_validation(X, y, n_splits, random_state, shuffle)
    else:
        X_train_folds, X_test_folds = kfold_cross_validation(X, n_splits, random_state, shuffle)

    y_true = []
    y_predicted = []

    for i in range(n_splits):
        X_train = [X[j] for j in X_train_folds[i]]
        X_test = [X[j] for j in X_test_folds[i]]
        y_train = [y[j] for j in X_train_folds[i]]
        y_test = [y[j] for j in X_test_folds[i]]

        classifier.fit(X_train, y_train)
        y_predictions = classifier.predict(X_test)
        y_true += y_test
        y_predicted += y_predictions

    matrix = confusion_matrix(y_true, y_predicted, labels)
    totals = [sum(row) for row in matrix]
    recog = []
    for i, row in enumerate(matrix):
        if sum(row) == 0:
            recog.append(0)
        else:
            recog.append(row[i] / sum(row))

    recog = [int(round(val, 2) * 100) for val in recog]
    matrix = [[labels[i]] + matrix[i] + [totals[i]] + [recog[i]] for i in range(len(matrix))]

    header = [label] + labels + ["Total", "Recognition (%)"]
    matrix = [header] + matrix

    return matrix

def binary_precision_score(y_true, y_pred, labels=None, pos_label=None):
    """Compute the precision (for binary classification). The precision is the ratio tp / (tp + fp)
        where tp is the number of true positives and fp the number of false positives.
        The precision is intuitively the ability of the classifier not to label as
        positive a sample that is negative. The best value is 1 and the worst value is 0.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of obj): The list of possible class labels. If None, defaults to
            the unique values in y_true
        pos_label(obj): The class label to report as the "positive" class. If None, defaults
            to the first label in labels

    Returns:
        precision(float): Precision of the positive class

    Notes:
        Loosely based on sklearn's precision_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html
    """
    if labels is None:
        labels = MyPyTable(["class"], y_true).get_atttribute_domain("class")

    if pos_label is None:
        pos_label = labels[0]

    def check_true_pos(i):
        if y_true[i] == pos_label and y_pred[i] == pos_label:
            return True
        else:
            return False

    def check_false_pos(i):
        if y_true[i] != pos_label and y_pred[i] == pos_label:
            return True
        else:
            return False

    true_pos = sum([1 for i in range(len(y_true)) if check_true_pos(i)])
    false_pos = sum([1 for i in range(len(y_true)) if check_false_pos(i)])

    # as long as there is one true positive, this won't be true
    if true_pos + false_pos == 0:
        bin_precision = 0.0
    else:
        bin_precision = true_pos / (true_pos + false_pos)

    return bin_precision

def binary_recall_score(y_true, y_pred, labels=None, pos_label=None):
    """Compute the recall (for binary classification). The recall is the ratio tp / (tp + fn) where tp is
        the number of true positives and fn the number of false negatives.
        The recall is intuitively the ability of the classifier to find all the positive samples.
        The best value is 1 and the worst value is 0.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of obj): The list of possible class labels. If None, defaults to
            the unique values in y_true
        pos_label(obj): The class label to report as the "positive" class. If None, defaults
            to the first label in labels

    Returns:
        recall(float): Recall of the positive class

    Notes:
        Loosely based on sklearn's recall_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html
    """
    if labels is None:
        labels = MyPyTable(["class"], y_true).get_atttribute_domain("class")

    if pos_label is None:
        pos_label = labels[0]

    def check_true_pos(i):
        if y_true[i] == pos_label and y_pred[i] == pos_label:
            return True
        else:
            return False

    def check_false_neg(i):
        if y_true[i] == pos_label and y_pred[i] != pos_label:
            return True
        else:
            return False

    true_pos = sum([1 for i in range(len(y_true)) if check_true_pos(i)])
    false_neg = sum([1 for i in range(len(y_true)) if check_false_neg(i)])

    # as long as there is one true positive, this won't be true
    if true_pos + false_neg == 0:
        bin_recall = 0.0
    else:
        bin_recall = true_pos / (true_pos + false_neg)

    return bin_recall

def binary_f1_score(y_true, y_pred, labels=None, pos_label=None):
    """Compute the F1 score (for binary classification), also known as balanced F-score or F-measure.
        The F1 score can be interpreted as a harmonic mean of the precision and recall,
        where an F1 score reaches its best value at 1 and worst score at 0.
        The relative contribution of precision and recall to the F1 score are equal.
        The formula for the F1 score is: F1 = 2 * (precision * recall) / (precision + recall)

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of obj): The list of possible class labels. If None, defaults to
            the unique values in y_true
        pos_label(obj): The class label to report as the "positive" class. If None, defaults
            to the first label in labels

    Returns:
        f1(float): F1 score of the positive class

    Notes:
        Loosely based on sklearn's f1_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
    """
    if labels is None:
        labels = MyPyTable(["class"], y_true).get_atttribute_domain("class")

    if pos_label is None:
        pos_label = labels[0]

    # handling labels and pos_label outside of functions to ensure they have same
    bin_precision = binary_precision_score(y_true, y_pred, labels, pos_label)
    bin_recall = binary_recall_score(y_true, y_pred, labels, pos_label)

    # if prec + recall == 0, then they are both 0 and top would be 0 anyways
    if bin_precision + bin_recall == 0:
        bin_f1_score = 0.0
    else:
        bin_f1_score = 2 * (bin_precision * bin_recall) / (bin_precision + bin_recall)

    return bin_f1_score

def print_bin_eval_with_folds(classifier, X_train_folds, X_test_folds, y_train_folds, y_test_folds, labels=None, pos_label=None):
    """Computes binary stats for a classifier given folds and creates a formatted confusion matrix

    Args:
        classifier(mysklearn classifier): classifier to be evaluated
        X_train_folds(nested list of objects): The list of training sets
        X_test_folds(nested list of objects): The list of testing set indices for each fold
        y_train(list of list of objects): the list of training set classes
        y_test(list of list of objects): the list of test set classes
        labels(list of obj): The list of possible class labels. If None, defaults to
            the unique values in y_true, in this function labels must be binary
            and the positive label is assumed to be first
        pos_label(obj): The class label to report as the "positive" class. If None, defaults
            to the first label in labels
    """
    y_predicted = []
    y_true = []
    for X_train, X_test, y_train, y_test in zip(X_train_folds, X_test_folds, y_train_folds, y_test_folds):
        classifier.fit(X_train, y_train)
        y_predicted += classifier.predict(X_test)
        y_true += y_test

    accuracy = round(accuracy_score(y_true, y_predicted), 2)
    error_rate = round(1 - accuracy, 2)
    precision = round(binary_precision_score(y_true, y_predicted, labels, pos_label), 2)
    recall = round(binary_recall_score(y_true, y_predicted, labels, pos_label), 2)
    f1 = round(binary_f1_score(y_true, y_predicted, labels, pos_label), 2)
    con_matrix = confusion_matrix(y_true, y_predicted, labels)
    con_matrix = [[labels[0]] + con_matrix[0], [labels[1]] + con_matrix[1]]
    con_matrix = [["class", labels[0], labels[1]]] + con_matrix

    print("accuracy =",  accuracy)
    print("error rate =", error_rate)
    print("precision =", precision)
    print("recall =", recall)
    print("f1 =", f1)
    print("Confusion Matrix")
    print(tabulate(con_matrix, headers="firstrow"))
