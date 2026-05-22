from flask import Flask, render_template, request

app = Flask(__name__)

@app.route("/")
def home():
    return render_template('Home.html')

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

@app.route('/LogisticRegressionConcepts', methods=['GET', 'POST'])
def logistic_regression_concepts():
    return render_template('LogisticRegression/LogisticRegressionConcepts.html')

@app.route('/LogisticRegressionApplication', methods=['GET', 'POST'])
def logistic_regression_application():
    return render_template('LogisticRegression/LogisticRegressionApplication.html')

if __name__ == "__main__":
    app.run(debug=True)