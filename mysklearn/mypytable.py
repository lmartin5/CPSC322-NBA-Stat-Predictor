"""CPSC 322 Final Project: NBA Team Success Predictor
@author L. Martin
@date March 30, 2022

mysklearn.mypytable.py
Description:
    This python file contains the MyPyTable class.
    It is similar to a Pandas DataFrame.
    See the class docstring for more info.
"""

import copy
import csv
from tabulate import tabulate

class MyPyTable:
    """Represents a 2D table of data with column names.

    Attributes:
        column_names(list of str): M column names
        data(list of list of obj): 2D data structure storing mixed type data.
            There are N rows by M columns.
    """

    def __init__(self, column_names=None, data=None):
        """Initializer for MyPyTable.

        Args:
            column_names(list of str): initial M column names (None if empty)
            data(list of list of obj): initial table data in shape NxM (None if empty)
        """
        if column_names is None:
            column_names = []
        self.column_names = copy.deepcopy(column_names)
        if data is None:
            data = []
        self.data = copy.deepcopy(data)

    def copy(self):
        """Creates a deepcopy of the MyPyTable.

        Returns:
            MyPyTable: a new MyPyTable object with deep copied column names and data
        """
        new_table = MyPyTable()
        new_table.column_names = copy.deepcopy(self.column_names)
        new_table.data = copy.deepcopy(self.data)
        return new_table

    def pretty_print(self):
        """Prints the table in a nicely formatted grid structure.
        """
        print(tabulate(self.data, headers=self.column_names))

    def __str__(self):
        """Gets a string rep of table as a nicely formatted grid structure.

        Returns:
            str: a nicely formatted string rep. of a MyPyTable from tabulate
        """
        return tabulate(self.data, headers=self.column_names)

    def get_shape(self):
        """Computes the dimension of the table (N x M).

        Returns:
            int: number of rows in the table (N)
            int: number of cols in the table (M)
        """
        return len(self.data), len(self.column_names)

    def get_column(self, col_identifier, include_missing_values=True):
        """Extracts a column from the table data as a list.

        Args:
            col_identifier(str or int): string for a column name or int
                for a column index
            include_missing_values(bool): True if missing values ("NA")
                should be included in the column, False otherwise.

        Returns:
            list of obj: 1D list of values in the column

        Notes:
            Raise ValueError on invalid col_identifier
        """
        # first checks if arg was a str or an int
        if isinstance(col_identifier, str):
            # checking to see if str given is one of the columns
            if col_identifier in self.column_names:
                col_index = self.column_names.index(col_identifier)
            else:
                raise ValueError("'" + col_identifier + "' is not a valid column name")
        else:
            col_index = col_identifier

        if include_missing_values:
            col = [row[col_index] for row in self.data]
        else:
            col = [row[col_index] for row in self.data if row[col_index] != "NA"]

        return col

    def get_atttribute_domain(self, col_identifier):
        """Extracts all seen values from a categorical/discrete column as a list.

        Args:
            col_identifier(str or int): string for a column name or int
                for a column index
            include_missing_values(bool): True if missing values ("NA")
                should be included in the column, False otherwise.

        Returns:
            list of obj: 1D list of unique values in the column
        """
        domain = set(self.get_column(col_identifier))
        domain = list(domain)
        domain.sort()
        return domain

    def convert_to_numeric(self):
        """Try to convert each value in the table to a numeric type (float).

        Notes:
            Leave values as is that cannot be converted to numeric.
        """
        for row in self.data:
            for col in range(len(self.column_names)):
                # will only convert data to float that can be converted
                try:
                    row[col] = float(row[col])
                except ValueError:
                    pass

    def convert_col_to_numeric(self, col_name):
        """Try to convert each value in the column to a numeric type (float).

        Args:
            col_name(str): name of column attempted to convert

        Notes:
            Leave values as is that cannot be converted to numeric.
        """
        col_index = self.column_names.index(col_name)

        for row in self.data:
            # will only convert data to float that can be converted
            try:
                row[col_index] = float(row[col_index])
            except ValueError:
                pass

    def drop_column(self, col_name):
        """Drops a column from MyPyTable

        Args:
            col_name(str): name of column to drop
        """
        col_index = self.column_names.index(col_name)
        new_data = []

        for row in self.data:
            new_row = [row[i] for i in range(len(self.column_names)) if i != col_index]
            new_data.append(new_row)

        self.data = new_data
        self.column_names = [self.column_names[i] for i in range(len(self.column_names)) if i != col_index]

    def make_table_from_list_attribute(self, list_attrib, key_col_names):
        """Return a new MyPyTable that is made up of the data from the list_attrib column
        but the values are now seperated into their own instances

        Args:
            list_attrib(MyPyTable): a column's name whose data is a comma seperated list
                rather than primitive type
            key_col_names(list of str): column names that are row keys, so the tables can be
                joined using perform_inner/outer_join later

        Returns:
            MyPyTable: the new table containing rows for all the data
        """
        new_column_names = key_col_names + [list_attrib]
        list_index = self.column_names.index(list_attrib)
        key_indexes = [self.column_names.index(name) for name in key_col_names]

        new_data = []
        for row in self.data:
            key_values = [row[key] for key in key_indexes]
            list_values = row[list_index]
            list_values = list_values.strip()
            list_values = list_values.split(",")
            for list_val in list_values:
                new_row = copy.deepcopy(key_values) + [list_val]
                new_data.append(new_row)

        created_table = MyPyTable()
        created_table.column_names = copy.deepcopy(new_column_names)
        created_table.data = new_data

        return created_table

    def convert_col_to_int(self, col_name):
        """Try to convert each value in the column to int type.

        Args:
            col_name(str): name of column attempted to convert

        Notes:
            Leave values as is that cannot be converted to int.
        """
        col_index = self.column_names.index(col_name)

        for row in self.data:
            # will only convert data to int that can be converted
            try:
                row[col_index] = int(row[col_index])
            except ValueError:
                pass

    def drop_rows(self, row_indexes_to_drop):
        """Remove rows from the table data.

        Args:
            row_indexes_to_drop(list of int): list of row indexes to remove from the table data.
        """
        # this is faster than using remove, which searches for the element every time it is called
        # just storing the elements found at the indexes not to be deleted
        data = [self.data[row] for row in range(len(self.data)) if row not in row_indexes_to_drop]
        self.data = data

    def load_from_file(self, filename, convert_to_numeric=True):
        """Load column names and data from a CSV file.

        Args:
            filename(str): relative path for the CSV file to open and load the contents of.
            convert_to_numeric(bool): True if should call self.convert_to_numeric() after load

        Returns:
            MyPyTable: return self so the caller can write code like
                table = MyPyTable().load_from_file(fname)

        Notes:
            Use the csv module.
            First row of CSV file is assumed to be the header.
        """
        with open(filename, "r") as infile:
            csv_read = csv.reader(infile, delimiter=',')
            rows = []

            for row in csv_read:
                rows.append(row)

            header = rows.pop(0)
            infile.close()

        self.column_names = header
        self.data = rows

        if convert_to_numeric:
            self.convert_to_numeric()

        return self

    def save_to_file(self, filename):
        """Save column names and data to a CSV file.

        Args:
            filename(str): relative path for the CSV file to save the contents to.

        Notes:
            Use the csv module.
        """
        with open(filename, "w") as outfile:
            csv_write = csv.writer(outfile, delimiter=',')

            csv_write.writerow(self.column_names)
            for row in self.data:
                csv_write.writerow(row)

            outfile.close()

    def find_duplicates(self, key_column_names):
        """Returns a list of indexes representing duplicate rows.
        Rows are identified uniquely based on key_column_names.

        Args:
            key_column_names(list of str): column names to use as row keys.

        Returns
            list of int: list of indexes of duplicate rows found

        Notes:
            Subsequent occurrence(s) of a row are considered the duplicate(s).
                The first instance of a row is not considered a duplicate.
        """
        cols = [self.get_column(key) for key in key_column_names]
        duplicate_indexes = []
        seen_keys = []

        for row in range(len(self.data)):
            # key stores the values in the key_columns for the row
            key = [cols[i][row] for i in range(len(cols))]
            if key in seen_keys:
                duplicate_indexes.append(row)
            else:
                seen_keys.append(key)

        return duplicate_indexes

    def get_duplicates(self, key_column_names):
        """Returns a list of the duplicate rows.
        Rows are identified uniquely based on key_column_names.

        Args:
            key_column_names(list of str): column names to use as row keys.

        Returns
            list of lists: list of the rows where duplicate keys found

        Notes:
            Subsequent occurrence(s) of a row are considered the duplicate(s).
                The first instance of a row is not considered a duplicate.
        """
        dup_indexes = self.find_duplicates(key_column_names)
        dups = [copy.deepcopy(self.data[index]) for index in dup_indexes]
        return dups

    def remove_duplicates(self, key_column_names):
        """Deletes all of the duplicate rows.
        Rows are identified uniquely based on key_column_names.

        Args:
            key_column_names(list of str): column names to use as row keys.

        Notes:
            Subsequent occurrence(s) of a row are considered the duplicate(s).
                The first instance of a row is not considered a duplicate.
        """
        self.drop_rows(self.find_duplicates(key_column_names))

    def remove_rows_with_missing_values(self):
        """Remove rows from the table data that contain a missing value ("NA").
        """
        row_index = 0
        while row_index < len(self.data):
            if "NA" in self.data[row_index]:
                self.data.pop(row_index)
            else:
                # only incrementing when not deleting to deal with index shift
                row_index += 1

    def replace_missing_values_with_column_average(self, col_name):
        """For columns with continuous data, fill missing values in a column
            by the column's original average.

        Args:
            col_name(str): name of column to fill with the original average (of the column).
        """
        col = self.get_column(col_name, False) # not including NA values to compute average
        col_average = sum(col) / len(col)
        col_index = self.column_names.index(col_name)

        for row in self.data:
            if row[col_index] == "NA":
                row[col_index] = col_average

    def replace_val_with_NA(self, value):
        """Replaces every instance of value in the dataset with "NA"
        Used if a dataset usees a different NA value, such as N/A

        Args:
            value(str): string value to be replaced with NA
        """
        for row in self.data:
            for i, _ in enumerate(row):
                if row[i] == value:
                    row[i] = "NA"

    def compute_summary_statistics(self, col_names):
        """Calculates summary stats for this MyPyTable and stores the stats in a new MyPyTable.

        Args:
            col_names(list of str): names of the continuous columns to compute summary stats for.

        Returns:
            MyPyTable: stores the summary stats computed. The column names and their order
                is as follows: ["attribute", "min", "max", "mid", "avg", "median"]
        """
        header = ["attribute", "min", "max", "mid", "avg", "median"]
        rows = []
        for col_name in col_names:
            # not using missing values if still in table to calc stats
            col = self.get_column(col_name, False)
            count = len(col)
            if count == 0:
                continue

            minimum = min(col)
            maximum = max(col)
            mid = (maximum + minimum) / 2
            avg = sum(col) / count

            sorted_col = sorted(col)
            if len(col) % 2 == 0:
                median = (sorted_col[count // 2] + sorted_col[count // 2 - 1]) / 2
            else:
                median = sorted_col[count // 2]

            rows.append([col_name, minimum, maximum, mid, avg, median])

        return MyPyTable(header, rows)

    def perform_inner_join(self, other_table, key_column_names):
        """Return a new MyPyTable that is this MyPyTable inner joined
            with other_table based on key_column_names.

        Args:
            other_table(MyPyTable): the second table to join this table with.
            key_column_names(list of str): column names to use as row keys.

        Returns:
            MyPyTable: the inner joined table.
        """
        cols_copy = copy.deepcopy(other_table.column_names)
        joined_header = copy.deepcopy(self.column_names)
        joined_header += [col for col in cols_copy if col not in key_column_names]
        joined_rows = []

        # indexes kept track of since they might not be in same order in both tables
        a_key_indexes = [self.column_names.index(col_name) for col_name in key_column_names]
        b_key_indexes = [other_table.column_names.index(col_name) for col_name in key_column_names]

        for a_row in self.data:
            # a_key stores the key values for a_row
            a_key = [a_row[col_index] for col_index in a_key_indexes]

            for b_row in other_table.data:
                # b_key stores the key values for b_row in the same order as a_key
                b_key = [b_row[col_index] for col_index in b_key_indexes]
                if a_key == b_key:
                    joined_row = copy.deepcopy(a_row)
                    # only adding the values for the columns that aren't key attrib.
                    joined_row += [b_row[i] for i in range(len(b_row)) if i not in b_key_indexes]
                    joined_rows.append(joined_row)

        return MyPyTable(joined_header, joined_rows)


    def perform_full_outer_join(self, other_table, key_column_names):
        """Return a new MyPyTable that is this MyPyTable fully outer joined with
            other_table based on key_column_names.

        Args:
            other_table(MyPyTable): the second table to join this table with.
            key_column_names(list of str): column names to use as row keys.

        Returns:
            MyPyTable: the fully outer joined table.

        Notes:
            Pad the attributes with missing values with "NA".
        """
        joined_header = copy.deepcopy(self.column_names)
        joined_header += [col for col in other_table.column_names if col not in key_column_names]
        joined_rows = []
        b_keys_with_matches = [] # used to keep track of which keys get matched from other_table

        a_key_indexes = [self.column_names.index(col_name) for col_name in key_column_names]
        b_key_indexes = [other_table.column_names.index(col_name) for col_name in key_column_names]

        for a_row in self.data:
            # a_key stores the key values for a_row
            a_key = [a_row[col_index] for col_index in a_key_indexes]
            a_match = False # stores if there was a match in other_table

            for b_row in other_table.data:
                # b_key stores the key values for b_row in the same order as a_key
                b_key = [b_row[col_index] for col_index in b_key_indexes]
                if a_key == b_key:
                    a_match = True
                    b_keys_with_matches.append(b_key)
                    joined_row = copy.deepcopy(a_row)
                    # only adding the values for the columns that aren't key attrib.
                    joined_row += [b_row[i] for i in range(len(b_row)) if i not in b_key_indexes]
                    joined_rows.append(joined_row)

            if not a_match:
                joined_row = copy.deepcopy(a_row)
                # filling other_table values with "NA"
                joined_row += ["NA"] * (len(other_table.column_names) - len(b_key_indexes))
                joined_rows.append(joined_row)

        # looking for keys that didn't have a match in other_table
        for b_row in other_table.data:
            # b_key stores the key values for b_row
            b_key = [b_row[col_index] for col_index in b_key_indexes]
            if b_key not in b_keys_with_matches:
                # filling self values with "NA"
                joined_row = ["NA"] * len(self.column_names)
                # replacing the "NA" in key values with key values found in b_row
                index = 0
                while index < len(a_key_indexes):
                    joined_row[a_key_indexes[index]] = b_key[index]
                    index += 1
                # only adding the values for the columns that aren't key attrib.
                joined_row += [b_row[i] for i in range(len(b_row)) if i not in b_key_indexes]
                joined_rows.append(joined_row)

        return MyPyTable(joined_header, joined_rows)

    def get_frequencies(self, col_identifier):
        """Return a dictionary where the keys are the unique values from col_name
        and the values are the frequencies that the values are seen in the instances.

        Args:
            col_identifier(str or int): string for a column name or int
                for a column index

        Returns:
            dict of ints: the number of times each unique value for col_name is seen in the table.

        Notes:
            Will include a count for all the rows with "NA" in the col_index position.
        """
        frequencies = {}
        column_vals = self.get_column(col_identifier)
        for val in column_vals:
            if val in frequencies.keys():
                frequencies[val] += 1
            else:
                frequencies[val] = 1

        return frequencies


    def groupby(self, col_identifier):
        """Return a dictionary of MyPyTables where the keys are the unique values
        from col_name and the values are MyPyTables with the instances
        containing those values in col_name position.

        Args:
            col_identifier(str or int): string for a column name or int
                for a column index

        Returns:
            dict of MyPyTables: the instances that contain each unique value for col_name.

        Notes:
            Will include a MyPyTable for all the rows with "NA" in the col_index position.
        """
        groupby_data = {label: [] for label in self.get_atttribute_domain(col_identifier)}

        # first checks if arg was a str or an int
        if isinstance(col_identifier, str):
            # checking to see if str given is one of the columns
            if col_identifier in self.column_names:
                col_index = self.column_names.index(col_identifier)
            else:
                raise ValueError("'" + col_identifier + "' is not a valid column name")
        else:
            col_index = col_identifier

        for row in self.data:
            groupby_data[row[col_index]].append(row)

        groupby_tables = {key: MyPyTable(copy.deepcopy(self.column_names), copy.deepcopy(data)) for key, data in groupby_data.items()}
        return groupby_tables
