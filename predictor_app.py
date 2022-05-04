# we are going to use the flask micro web framework
# for our web app (running on API service)
import pickle
from mysklearn.mypytable import MyPyTable
from flask import Flask, jsonify, request, render_template, redirect

import player_prep

# create our web app
# flask runs by default on port 5000
app = Flask(__name__)

# we are now ready to setup our first "route"
# route is a function that handles a request
@app.route('/', methods = ['GET', 'POST'])
def index():
    prediction = ""
    if request.method == "POST":
        pg = request.form["pg"]
        sg = request.form["sg"]
        sf = request.form["sf"]
        pf = request.form["pf"]
        c = request.form["c"]
        pg_szn = int(request.form["pg_szn"])
        sg_szn = int(request.form["sg_szn"])
        sf_szn = int(request.form["sf_szn"])
        pf_szn = int(request.form["pf_szn"])
        c_szn = int(request.form["c_szn"])
        instance = [[pg, pg_szn], [sg, sg_szn], [sf, sf_szn], [pf, pf_szn], [c, c_szn]]
        prediction = make_prediction(instance)
    print("prediction:", prediction)
    # goes into templates folder and finds given name
    return render_template("index.html", prediction=prediction)

@app.route("/predict", methods = ['GET', 'POST'])
def predict():
    # parse the query string to get our
    # instance attribute values from the client
    # paramNames are case sensitive

    pg = request.args.get("pg", "") # "" is default
    sg = request.args.get("sg", "") # "" is default
    sf = request.args.get("sf", "") # "" is default
    pf = request.args.get("pf", "") # "" is default
    c = request.args.get("c", "") # "" is default
    pg_szn = request.form["pg_szn"]
    sg_szn = request.form["sg_szn"]
    sf_szn = request.form["sf_szn"]
    pf_szn = request.form["pf_szn"]
    c_szn = request.form["c_szn"]

    instance = [[pg, pg_szn], [sg, sg_szn], [sf, sf_szn], [pf, pf_szn], [c, c_szn]]
    prediction = make_prediction(instance)
    # if anything goes wrong in this function, it will return none
    if prediction is not None:
        result = {"prediction": prediction}
        return jsonify(result), 200
    return "Error making prediction", 400

def make_prediction(instance):
    # we need a ML model to make a prediction for instance
    # typically the model in trained "offline"
    # and used later "online" (e.g. via this web app)
    # enter pickling
    # unpickle tree.p
    infile = open("player_stats.p", "rb")
    players = pickle.load(infile)
    infile.close()

    infile = open("knn.p", "rb")
    knn = pickle.load(infile)
    infile.close()

    team_data = []

    # try: 
    for player in instance:
        name = player[0]
        szn = player[1]
        if name not in players.keys():
            return -1
        player_data = players[name]
        if szn not in player_data.keys():
            return -2
        player_season_data = player_data[szn]

        team_index = player_season_data.column_names.index("Team")
        if len(player_season_data.data) == 1:
            team_data.append(player_season_data.data[0])
        else:
            for row in player_season_data.data:
                if row[team_index] == "Total":
                    team_data.append(row)
                    break
    custom_team = MyPyTable(player_season_data.column_names, team_data)
    team_table = player_prep.create_team_data(custom_team)
    

    team_table.drop_column("Team")
    team_table.drop_column("Season")

    prediction = knn.predict(team_table.data)[0]
    print(prediction)
    return prediction
    #except:
    #    print("error")
    #   return None

if __name__ == "__main__":
    # deployment notes
    # we need to get flask app on web
    # we can set up and maintain our own server or use cloud provider
    # (AWS, GCP, Azure, DigitalOcean, Heroku, ...)
    # we are going to use Heroku (PaaS platform as a service)
    # there are a few ways (4) to deploy a flask app to heroku (see my youtube videos)
    # today, 2.b.

    app.run(debug=True, port=5001) # TODO: turn debug off
    # when deploy to "production"