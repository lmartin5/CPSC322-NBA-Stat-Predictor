import os
import pickle # a standard library
# pickling is used for object serialization and deserialization
# pickle: write a binary represenation of an object to a file (for use later...)
# un/depickle: read a binary representation of an object from a file (to a python object in this memory)

from mysklearn.myclassifiers import MyKNeighborsClassifier
from mysklearn.mypytable import MyPyTable


file_loc = os.path.join("input_data", "processed_data", "team_stats.csv")
team_stats = MyPyTable().load_from_file(file_loc)

y = team_stats.get_column("Success")
team_stats.drop_column("Success")
team_stats.drop_column("Team")
team_stats.drop_column("Season")
X = team_stats.data

knn = MyKNeighborsClassifier(10)
knn.fit(X, y)

outfile = open("knn.p", "wb")
pickle.dump(knn, outfile)
outfile.close()