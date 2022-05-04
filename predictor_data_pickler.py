import os
import pickle # a standard library
# pickling is used for object serialization and deserialization
# pickle: write a binary represenation of an object to a file (for use later...)
# un/depickle: read a binary representation of an object from a file (to a python object in this memory)

from mysklearn.myclassifiers import MyDecisionTreeClassifier, MyKNeighborsClassifier, MyNaiveBayesClassifier
from mysklearn.mypytable import MyPyTable
from mysklearn.myutils import groupby

file_loc = os.path.join("input_data", "processed_data", "team_stats.csv")
team_stats = MyPyTable().load_from_file(file_loc)

y = team_stats.get_column("Success")
team_stats.drop_column("Success")
team_stats.drop_column("Team")
team_stats.drop_column("Season")
X = team_stats.data

knn = MyKNeighborsClassifier(10)
knn.fit(X, y)

nb = MyNaiveBayesClassifier()
nb.fit(X, y)

outfile = open("knn.p", "wb")
pickle.dump(knn, outfile)
outfile.close()

outfile = open("nb.p", "wb")
pickle.dump(nb, outfile)
outfile.close()

file_loc = os.path.join("input_data", "processed_data", "player_stats.csv")
player_stats = MyPyTable().load_from_file(file_loc)
players = player_stats.groupby("Player", "Season")

outfile = open("player_stats.p", "wb")
pickle.dump(players, outfile)
outfile.close()