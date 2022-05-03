import operator
import random # used to resolve equal probabilities in Naive Bayes
from math import log2
from copy import deepcopy

from mysklearn import myutils
from mysklearn.mypytable import MyPyTable

class MyRandomForestClassifier:
    """Represents a randomly generated group of decision tree classifiers.

    Attributes:
        X_remainder(list of list of obj): The list of training instances to split into
                train and validation sets.
                The shape of X_train is (n_train_samples, n_features)
        y_remainder(list of obj): The target y values (parallel to X_remainder).
            The shape of y_remainder is n_samples
        trees(list of trees): The extracted tree model.
        N(int): number of trees to generate
        M(int): number of "best" trees to keep
        F(int): number of randomly selected attributes to fit tree on

    Notes:
        Loosely based on sklearn's DecisionTreeClassifier:
            https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
        Terminology: instance = sample = row and attribute = feature = column
    """
    def __init__(self, N = 50, M = 5, F = 3):
        """Initializer for MyDecisionTreeClassifier.
        """
        self.X_remainder = None
        self.y_remainder = None
        self.trees = None
        self.N = N
        self.M = M
        self.F = F

    def fit(self, X_remainder, y_remainder):
        """Fits a random forest classifier to X_train and y_train using the TDIDT
        (top down induction of decision tree) algorithm.

        Args:
            X_remainder(list of list of obj): The list of training instances to split into
                    train and validation sets.
                    The shape of X_train is (n_train_samples, n_features)
            y_remainder(list of obj): The target y values (parallel to X_remainder).
                The shape of y_remainder is n_samples

        Notes:
            Since TDIDT is an eager learning algorithm, this method builds a decision tree model
                from the training data.
            Build a decision tree using the nested list representation described in class.
            On a majority vote tie, choose first attribute value based on attribute domain ordering.
            Store the tree in the tree attribute.
            Use attribute indexes to construct default attribute names (e.g. "att0", "att1", ...).
        """
        self.X_remainder = X_remainder
        self.y_remainder = y_remainder
        avail_attributes = ["att" + str(i) for i in range(len(X_remainder[0]))]

        # storing data in MyPyTable object for access to functions there
        header = avail_attributes + ["class"]
        train_data = [X_remainder[i] + [y_remainder[i]] for i in range(len(X_remainder))]
        train = MyPyTable(header, train_data)

        attrib_domains = {}
        for attribute in header:
            attrib_domains[attribute] = train.get_atttribute_domain(attribute)

        trees = []
        for i in range(self.N):
            X_train, X_valid, y_train, y_valid = bootstrap_sample(X_remainder, y_remainder)
            tree = MyDecisionTreeClassifier()
            random_attributes = self.get_random_attributes(avail_attributes)
            tree.fit(X_train, y_train, random_attributes, attrib_domains)
            y_predicted = tree.predict(X_valid)
            accuracy = accuracy_score(y_valid, y_predicted)
            trees.append([tree, accuracy])
        trees.sort(key=operator.itemgetter(1), reverse=True) # sorts in descending order
        self.trees = trees[:self.M]
        self.trees = [self.trees[i][0] for i in range(len(self.trees))]

    def get_random_attributes(self, attributes):
        """Gets a random subset of size F of the attributes to create a tree from.

        Args:
            attributes(list of strs): The list of attributtes

        Returns:
            attr(list of str): list of attributes of size self.F
        """
        attributes = attributes.copy()
        attr = []
        for i in range(self.F):
            rand_attrib = random.choice(attributes)
            attr.append(rand_attrib)
            attributes.remove(rand_attrib)
        return attr

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
            predictions = []
            for tree in self.trees:
                predictions.append(tree.predict([instance])[0])
                pred_freqs = myutils.get_frequencies(predictions)
                pred_freqs = [[pred, count] for pred, count in pred_freqs.items()]
                pred_freqs.sort(key=operator.itemgetter(1), reverse=True) # sorts in descending order
            y_predicted.append(pred_freqs[0][0])
        return y_predicted

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

    def fit(self, X_train, y_train, avail_attributes, attrib_domains=None):
        """Fits a decision tree classifier to X_train and y_train using the TDIDT
        (top down induction of decision tree) algorithm.

        Args:
            X_train(list of list of obj): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples
            avail_attributes(list of str): the list of attributes available to split on

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

        # storing data in MyPyTable object for access to functions there
        header = ["att" + str(i) for i in range(len(X_train[0]))] + ["class"]
        train_data = [X_train[i] + [y_train[i]] for i in range(len(X_train))]
        train = MyPyTable(header, train_data)

        self.attribute_domains = attrib_domains
        if attrib_domains is None:
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
        sampled_indexes.append(random.randint(0, len(X) - 1))
    X_sample = [deepcopy(X[i]) for i in sampled_indexes]
    X_out_of_bag = [deepcopy(X[i]) for i in range(len(X)) if i not in sampled_indexes]
    if y_vals:
        y_sample = [deepcopy(y[i]) for i in sampled_indexes]
        y_out_of_bag = [deepcopy(y[i]) for i in range(len(X)) if i not in sampled_indexes]

    return X_sample, X_out_of_bag, y_sample, y_out_of_bag

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