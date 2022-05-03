# we are going to use the flask micro web framework
# for our web app (running on API service)
import pickle
from flask import Flask, jsonify, request, render_template, redirect

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
        # prediction = make_prediction([pg, sg, sf, pf, c])
        prediction = "GOATED"
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

    instance = [pg, sg, sf, pf, c]
    print("attribute vals:", instance)

    # prediction = make_prediction(instance)
    prediction = "GOATED"
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
    infile = open("knn.p", "rb")
    knn = pickle.load(infile)
    infile.close()

    print(knn)

    try: 
        prediction = knn.predict([instance])[0]
        return prediction
    except:
        print("error")
        return None

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