from flask import Flask, render_template, request
from ClimateClustering import ClimateKMeans

app = Flask(__name__)


#-------------------- HOME --------------------
@app.route("/")
def home():
    return render_template('Home.html')

#-------------------- PAGES --------------------

@app.route("/businessunderstanding")
def business_data():
    return render_template('Businessunderstanding.html')

@app.route("/dataunderstanding")
def data_understanding():
    return render_template('dataunderstanding.html')

@app.route("/dataengineering")
def data_engineering():
    return render_template('DataEngineering.html')

@app.route("/crispmlmethodology")
def crisp_ml_methodology():
    return render_template('crispml.html')

if __name__ == "__main__":    app.run(debug=True)





# -------------------- CLIMATE KMEANS --------------------

@app.route("/Kmeans/concepts")
def kmeans_concepts():
    return render_template("Kmeans/conceptskmeans.html")

@app.route("/Kmeans/application")
def kmeans_application():

    k = request.args.get("k", default=3, type=int)

    info = ClimateKMeans(k)

    return render_template(
        "Kmeans/applicationkmeans.html",

        summary=info["summary"],

        centers=enumerate(info["centers"]),

        results=info["results"],

        graph=info["graph"],

        score=info["score"],

        inertia=info["inertia"],

        elbow_graph=info["elbow_graph"],

        k=k
    )

@app.route("/Kmeans/assessment")
def kmeans_assessment():
    return render_template("Kmeans/kmeans_assessment.html")


if __name__ == "__main__":
    app.run(debug=True)