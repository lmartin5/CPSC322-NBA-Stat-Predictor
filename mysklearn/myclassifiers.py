"""PA7
Name: Luke Martin
Clss: CPSC 322 - 02
Sems: Spring 2022
Date: April 12, 2022

mysklearn.myclassifers.py
Description:
    This python file contains the simple classifers for mysklearn:
        MySimpleLinearRegressor
        MyKNeighborsClassifier
        MyDummyClassifier
        MyNaiveBayesClassifer
        MyDecisionTreeClassifier
"""

import operator
import random # used to resolve equal probabilities in Naive Bayes
from math import log2

from mysklearn import myutils
from mysklearn.mypytable import MyPyTable
from mysklearn.mysimplelinearregressor import MySimpleLinearRegressor

# myclassifiers.py solution from PA4-6 below MyDecisionTree

class MyDecisionTreeClassifier:
    """Represents a decision tree classifier.

    Attributes:
        X_train(list of list of obj): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train).
            The shape of y_train is n_samples
        tree(nested list): The extracted tree model.
        rules(list of strs): holds the rules for when being recursively generated
        attribute_domains(dict of attr_name, lists): holds the domains for every attribute
            when tree being generated
        most_recent_attribute_size(int): used for case 3 in tree generation

    Notes:
        Loosely based on sklearn's DecisionTreeClassifier:
            https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
        Terminology: instance = sample = row and attribute = feature = column
    """
    def __init__(self):
        """Initializer for MyDecisionTreeClassifier.
        """
        self.X_train = None
        self.y_train = None
        self.attribute_domains = None
        self.tree = None
        self.rules = None
        self.most_recent_attribute_size = 0

    def fit(self, X_train, y_train):
        """Fits a decision tree classifier to X_train and y_train using the TDIDT
        (top down induction of decision tree) algorithm.

        Args:
            X_train(list of list of obj): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since TDIDT is an eager learning algorithm, this method builds a decision tree model
                from the training data.
            Build a decision tree using the nested list representation described in class.
            On a majority vote tie, choose first attribute value based on attribute domain ordering.
            Store the tree in the tree attribute.
            Use attribute indexes to construct default attribute names (e.g. "att0", "att1", ...).
        """
        self.X_train = X_train
        self.y_train = y_train
        avail_attributes = ["att" + str(i) for i in range(len(X_train[0]))]

        # storing data in MyPyTable object for access to functions there
        header = avail_attributes + ["class"]
        train_data = [X_train[i] + [y_train[i]] for i in range(len(X_train))]
        train = MyPyTable(header, train_data)

        self.attribute_domains = {}
        for attribute in header:
            self.attribute_domains[attribute] = train.get_atttribute_domain(attribute)

        self.most_recent_attribute_size = len(self.X_train)
        self.tree = self.tdidt(train, avail_attributes)

    def tdidt(self, table, avail_attributes):
        """Fits a decision tree classifier to X_train and y_train using the TDIDT
        (top down induction of decision tree) algorithm.

        Args:
            table(MyPyTable): the data that is currently available to tree
                will be original data or partition of original data
            avail_attributes(list of strs): the attributes that we can select
                and hence have not yet been split on

        Returns:
            tree(list of lists): subtree structure for the data, final tree returned
                will be stored as the model for this class
        """
        attribute = self.select_attribute(table, avail_attributes)
        avail_attributes.remove(attribute) # can't split on again below this node
        tree = ["Attribute", attribute] # starting to build the tree

        data_size = len(table.data)
        partitions = self.partition_instances(table, attribute)
        for att_value, att_partition in partitions.items():
            value_subtree = ["Value", att_value]

            # CASE 1:
            if len(att_partition.data) > 0 and self.check_all_same_class(att_partition):
                # Making a leaf node
                class_label = att_partition.data[0][-1] # choice of row arbitrary if all same
                label_count = len(att_partition.data)
                value_subtree.append(["Leaf", class_label, label_count, data_size])

            # CASE 2:
            elif len(att_partition.data) > 0 and len(avail_attributes) == 0:
                # no more attributes to select (clash)
                # on tie pick one first in alphabetical order
                class_distribution = att_partition.get_frequencies("class")
                most_common_class_label = "none" # will get overwritten in first loop iteration
                most_common_count = 0
                for class_label, count in class_distribution.items():
                    if count >= most_common_count:
                        if count > most_common_count:
                            most_common_count = count
                            most_common_class_label = class_label
                        elif class_label < most_common_class_label:
                            # goes here on tie, when label comes first alphabetically
                            most_common_class_label = class_label

                label_count = len(att_partition.data)
                value_subtree.append(["Leaf", most_common_class_label, most_common_count, data_size])

            # CASE 3:
            elif len(att_partition.data) == 0:
                # "backtrack" and replace this attribute node with a
                # majority vote leaf node
                labels = table.get_frequencies("class") # using MyPyTable() function
                most_common_class_label = "none" # will get overwritten in first loop iteration
                most_common_count = 0
                for class_label, count in labels.items():
                    if count >= most_common_count:
                        if count > most_common_count:
                            most_common_count = count
                            most_common_class_label = class_label
                        elif class_label < most_common_class_label:
                            # goes here on tie, when label comes first alphabetically
                            most_common_class_label = class_label
                tree = ["Leaf", most_common_class_label, len(table.data), self.most_recent_attribute_size]
                return tree

            # NO CASES
            else: # none of the previous conditions were true... recurse!
                self.most_recent_attribute_size = len(table.data)
                subtree = self.tdidt(att_partition, avail_attributes.copy())
                value_subtree.append(subtree)
            tree.append(value_subtree)

        return tree

    def partition_instances(self, table, split_attribute):
        """Partitions the table of data based on split_attribute
        Different than groupby since empty partitions are returned too.
        Based on domain made in fit()

        Args:
            table(MyPyTable): the data that is currently available to tree
                will be original data or partition of original data
            split_attribute(str): the attribute to partition on

        Returns:
            partitions(dict of val, MyPyTable pairs): the partitions for each value in domain
        """
        att_index = table.column_names.index(split_attribute)
        att_domain = self.attribute_domains[split_attribute]
        parts = {domain_val: [].copy() for domain_val in att_domain}

        for row in table.data:
            parts[row[att_index]].append(row)

        partitions = {}
        for val, part in parts.items():
            # deep copying data not needed since we aren't doing any manipulation
            partitions[val] = MyPyTable(table.column_names.copy(), part.copy())

        return partitions

    def check_all_same_class(self, table):
        """determines if the class label is the same for all
        instances in the table data

        Args:
            table(MyPyTable): the data that is currently available to tree
                will be original data or partition of original data

        Returns:
            all_in(bool): True if all same class, else False
        """
        first_class = table.data[0][-1] # class is always in last column
        for row in table.data:
            if row[-1] != first_class:
                return False
        return True

    def select_attribute(self, table, avail_attributes):
        """Selects an attribute for node creatation based on entropy.
        Calculates e_new for every attribute and selects attribute
        that leads to highest information gain

        Args:
            table(MyPyTable): the data that is currently available to tree
                will be original data or partition of original data
            avail_attributes(list of strs): the attributes that we can select
                and hence entropy calculated for

        Returns:
            attribute(list of obj): the attribute that leads to most info gain
        """
        e_news = {}
        num_instances = len(table.data)
        # using entropy formula from entropy lab, done for each attribute
        for att in avail_attributes:
            attrib_partitions = self.partition_instances(table, att)
            value_entropys = []
            partition_sizes = []
            # calculating entropy for each value in attribute using domain
            for partition in attrib_partitions.values():
                partition_size = len(partition.data)
                partition_sizes.append(partition_size)
                class_partitions = self.partition_instances(partition, "class")
                entropy_vals = []
                for class_part in class_partitions.values():
                    class_count = len(class_part.data)
                    if class_count == 0: # can't take natural log of a number, will be 0 if part_size is too
                        entropy_vals.append(0)
                    else:
                        proportion = class_count / partition_size
                        entropy_vals.append(-1 * proportion * log2(proportion))
                value_entropys.append(sum(entropy_vals))
            e_new = sum([entr * (count / num_instances) for entr, count in zip(value_entropys, partition_sizes)])
            e_news[att] = e_new

        attrib_entropies = [[attrib, entrop] for attrib, entrop in e_news.items()]
        attrib_entropies.sort(key=operator.itemgetter(1)) # sorts in ascending order
        return attrib_entropies[0][0] # attribute with lowest e_new, and thus highest info gain


    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        y_predicted = []
        for instance in X_test:
            y_predicted.append(self.tdidt_predict(self.tree, instance))
        return y_predicted

    def tdidt_predict(self, tree, instance):
        """Makes prediction recursively for an instance.

        Args:
            tree(list of lists): subtree structure to be descended down, leaf node base case
            instance(list of objects): one instance from data to make prediction on

        Returns:
            prediction (obj): the prediction for the instance gotten at leaf level
        """
        info_type = tree[0]
        if info_type == "Leaf":
            return tree[1] # location of our prediction

        # else, we will be in an attribute node by setup
        att_name = tree[1]
        # will always consist of 'attx' where x is a non-negative int
        att_index = int(att_name[3:])

        for i in range(2, len(tree)):
            value_subtree = tree[i]
            if value_subtree[1] == instance[att_index]:
                # MATCH! RECURSE!
                return self.tdidt_predict(value_subtree[2], instance)

    def print_decision_rules(self, attribute_names=None, class_name="class"):
        """Prints the decision rules from the tree in the format
        "IF att == val AND ... THEN class = label", one rule on each line.

        Args:
            attribute_names(list of str or None): A list of attribute names to use in the decision rules
                (None if a list is not provided and the default attribute names based on indexes
                (e.g. "att0", "att1", ...) should be used).
            class_name(str): A string to use for the class name in the decision rules
                ("class" if a string is not provided and the default name "class" should be used).
        """
        self.rules = [] # holds the rule strings
        self.tdidt_get_rules("IF ", self.tree, attribute_names, class_name)
        for rule in self.rules:
            print(rule)

    def tdidt_get_rules(self, curr_rule, tree, attribute_names, class_name):
        """Recursively descends the tree to store string versions of the rules
        determined by the tree structure in the form:
        "IF att == val AND ... THEN class = label", one rule on each line.

        Args:
            curr_rule(str): holds the rule string, and is added onto recursively
            tree(list of lists): subtree structure to be descended down, leaf node base case
            attribute_names(list of str or None): A list of attribute names to use in the decision rules
                (None if a list is not provided and the default attribute names based on indexes
                (e.g. "att0", "att1", ...) should be used).
            class_name(str): A string to use for the class name in the decision rules
                ("class" if a string is not provided and the default name "class" should be used).
        """
        info_type = tree[0]

        if info_type == "Leaf":
            # erasing the most recently added 'AND ' (not needed on end)
            curr_rule = curr_rule[:-4]
            curr_rule += "THEN " + str(class_name) + " = " + str(tree[1])
            self.rules.append(curr_rule)

        if info_type == "Attribute":
            att_name = tree[1]
            if attribute_names is not None:
                # will always consist of 'attx' where x is a non-negative int, if auto gen
                att_index = int(att_name[3:])
                att_name = attribute_names[att_index]
            curr_rule += str(att_name) + " == "

            for i in range(2, len(tree)):
                # RECURSE
                value_subtree = tree[i]
                self.tdidt_get_rules(curr_rule, value_subtree, attribute_names, class_name)

        if info_type == "Value":
            value = tree[1]
            curr_rule += str(value) + " AND "
            # RECURSE
            self.tdidt_get_rules(curr_rule, tree[2], attribute_names, class_name)

    # BONUS method - DID NOT IMPLEMENT
    def visualize_tree(self, dot_fname, pdf_fname, attribute_names=None):
        """BONUS: Visualizes a tree via the open source Graphviz graph visualization package and
        its DOT graph language (produces .dot and .pdf files).

        Args:
            dot_fname(str): The name of the .dot output file.
            pdf_fname(str): The name of the .pdf output file generated from the .dot file.
            attribute_names(list of str or None): A list of attribute names to use in the decision rules
                (None if a list is not provided and the default attribute names based on indexes
                (e.g. "att0", "att1", ...) should be used).

        Notes:
            Graphviz: https://graphviz.org/
            DOT language: https://graphviz.org/doc/info/lang.html
            You will need to install graphviz in the Docker container as shown in class to complete this method.
        """
        pass

class MySimpleLinearRegressionClassifier:
    """Represents a simple linear regression classifier that discretizes
        predictions from a simple linear regressor (see MySimpleLinearRegressor).

    Attributes:
        discretizer(function): a function that discretizes a numeric value into
            a string label. The function's signature is func(obj) -> obj
        regressor(MySimpleLinearRegressor): the underlying regression model that
            fits a line to x and y data

    Notes:
        Terminology: instance = sample = row and attribute = feature = column
    """

    def __init__(self, discretizer, regressor=None):
        """Initializer for MySimpleLinearClassifier.

        Args:
            discretizer(function): a function that discretizes a numeric value into
                a string label. The function's signature is func(obj) -> obj
            regressor(MySimpleLinearRegressor): the underlying regression model that
                fits a line to x and y data (None if to be created in fit())
        """
        self.discretizer = discretizer
        self.regressor = regressor

    def fit(self, X_train, y_train):
        """Fits a simple linear regression line to X_train and y_train.

        Args:
            X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples
        """
        self.regressor = MySimpleLinearRegressor()
        self.regressor.fit(X_train, y_train)

    def predict(self, X_test):
        """Makes predictions for test samples in X_test by applying discretizer
            to the numeric predictions from regressor.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        y_predicted = self.regressor.predict(X_test)
        y_predicted = [self.discretizer(y) for y in y_predicted]
        return y_predicted

class MyKNeighborsClassifier:
    """Represents a simple k nearest neighbors classifier.

    Attributes:
        n_neighbors(int): number of k neighbors
        X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train).
            The shape of y_train is n_samples

    Notes:
        Loosely based on sklearn's KNeighborsClassifier:
            https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
        Terminology: instance = sample = row and attribute = feature = column
        Assumes data has been properly normalized before use.
    """
    def __init__(self, n_neighbors=3):
        """Initializer for MyKNeighborsClassifier.

        Args:
            n_neighbors(int): number of k neighbors
        """
        self.n_neighbors = n_neighbors
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        """Fits a kNN classifier to X_train and y_train.

        Args:
            X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since kNN is a lazy learning algorithm, this method just stores X_train and y_train
        """
        self.X_train = X_train
        self.y_train = y_train

    def kneighbors(self, X_test):
        """Determines the k closest neighbors of each test instance.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            distances(list of list of float): 2D list of k nearest neighbor distances
                for each instance in X_test
            neighbor_indices(list of list of int): 2D list of k nearest neighbor
                indices in X_train (parallel to distances)
        """
        distances = []
        neighbor_indices = []
        for X in X_test:
            dists = [[i, myutils.dist(X, self.X_train[i])] for i in range(len(self.X_train))]
            dists.sort(key=operator.itemgetter(1)) # sort by distance in asc order

            distances.append([dists[i][1] for i in range(self.n_neighbors)])
            neighbor_indices.append([dists[i][0] for i in range(self.n_neighbors)])
        return distances, neighbor_indices

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        y_predicted = []
        _, predicted_neighbors = self.kneighbors(X_test)
        for neighbors in predicted_neighbors:
            neighbors_classes = [self.y_train[n] for n in neighbors]
            classes_frequency = myutils.get_frequencies(neighbors_classes)
            classes_frequency = [[val, freq] for val, freq in classes_frequency.items()]
            classes_frequency.sort(key=operator.itemgetter(1)) # sort by freq
            prediction = classes_frequency[-1][0] # stores most common label
            y_predicted.append(prediction)
        return y_predicted


class MyDummyClassifier:
    """Represents a "dummy" classifier using the "most_frequent" strategy.
        The most_frequent strategy is a Zero-R classifier, meaning it ignores
        X_train and produces zero "rules" from it. Instead, it only uses
        y_train to see what the most frequent class label is. That is
        always the dummy classifier's prediction, regardless of X_test.

    Attributes:
        most_common_label(obj): whatever the most frequent class label in the
            y_train passed into fit()

    Notes:
        Loosely based on sklearn's DummyClassifier:
            https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html
    """
    def __init__(self):
        """Initializer for DummyClassifier.

        """
        self.most_common_label = None

    def fit(self, X_train, y_train):
        """Fits a dummy classifier to X_train and y_train.

        Args:
            X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since Zero-R only predicts the most frequent class label, this method
                only saves the most frequent class label.
        """
        freqs = myutils.get_frequencies(y_train)
        freqs = [[val, freq] for val, freq in freqs.items()]
        freqs.sort(key=operator.itemgetter(1)) # sort by freq
        self.most_common_label = freqs[-1][0] # stores most common label

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        return [self.most_common_label] * len(X_test)

class MyNaiveBayesClassifier:
    """Represents a Naive Bayes classifier.

    Attributes:
        priors(dict of value, probability pairs): The prior probabilities computed for each
            class label in the training set.
        posteriors(dict of value, (dict of attribute, probability pairs) pairs): The posterior probabilities computed for each
            attribute value/label pair in the training set. The keys in the outer dictionary are each of the possible class labels
            given the data in y_train. The values of the outer dictionary are lists containing a dictionary for each attribute in the
            data set. The keys in the inner dictionary are one for each possible value for the attribute the dictionary corresponds to.
            Then the value for the inner dictionary represents the posterior probability.
            Ex. self.posteriors["yes"][0]["maybe"] represents the probability that an instance has the value "maybe" in its first attribute
            given that its class label is "yes"

    Notes:
        Loosely based on sklearn's Naive Bayes classifiers: https://scikit-learn.org/stable/modules/naive_bayes.html
        You may add additional instance attributes if you would like, just be sure to update this docstring
        Terminology: instance = sample = row and attribute = feature = column
    """
    def __init__(self, priors=None, posteriors=None):
        """Initializer for MyNaiveBayesClassifier.

        Arguments:
            priors(dict of value, probability pairs): The prior probabilities computed for each
                label in the training set.
            posteriors(dict of value, (list of dict of value, probability pairs) pairs): The posterior probabilities computed for each
                attribute value/label pair in the training set.
        """
        self.priors = priors
        self.posteriors = posteriors

    def fit(self, X_train, y_train):
        """Fits a Naive Bayes classifier to X_train and y_train.

        Args:
            X_train(list of list of obj): The list of training instances (samples)
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples
        """
        num_attributes = len(X_train[0])
        train_header = list(range(num_attributes)) + ["class"]
        train_data = [X_train[i] + [y_train[i]] for i in range(len(X_train))]
        train = MyPyTable(train_header, train_data)

        class_domain = train.get_atttribute_domain("class")
        self.priors = {label: 0.0 for label in class_domain}
        self.posteriors = {label: [] for label in class_domain}

        num_instances = len(train.data)
        class_splits = train.groupby("class")
        for label in class_domain:
            self.priors[label] = len(class_splits[label].data) / num_instances

        # iterating through each attribute
        for index in range(num_attributes):
            attribute_domain = train.get_atttribute_domain(index)
            for label in class_domain:
                self.posteriors[label].append({attrib: 0.0 for attrib in attribute_domain})
                subset = class_splits[label]
                attribute_cts = subset.get_frequencies(index)
                # only updating attribute probabilities when not equal to 0
                for attrib_value in list(attribute_cts.keys()):
                    prob = attribute_cts[attrib_value] / len(subset.data)
                    self.posteriors[label][index][attrib_value] = prob

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        y_pred = []

        # iterating through every instance in X_test
        for instance in X_test:
            max_prob = -1 # guarentees that there is at least one max
            probabilities = {} # stores the class label \ probability pairs for max probs

            # calculates the prob for each class label using Bayes' Theorem,
            # adding to probabilties if higher than max_prob
            for class_label in self.priors.keys():
                curr_prob = self.priors[class_label]

                # grabbing the posterior for each attribute value in the instance
                for i, attrib_prob_dict in enumerate(self.posteriors[class_label]):
                    value_prob = attrib_prob_dict[instance[i]]
                    curr_prob = curr_prob * value_prob # updating prob by alg def

                if curr_prob > max_prob:
                    probabilities = {class_label: curr_prob}
                    max_prob = curr_prob
                elif curr_prob == max_prob:
                    probabilities[class_label] = curr_prob

            # if there are ties for highest probability, choose a random label
            y_pred.append(random.choice(list(probabilities.keys()))) # selects the lone key when only one as usual

        return y_pred
