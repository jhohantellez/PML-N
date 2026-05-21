from flask import Flask, render_template, request

app = Flask(__name__)


#-------------------- HOME --------------------
@app.route("/")
def home():
    return render_template('Home.html')
#-------------------- PAGES --------------------

@app.route("/businessunderstanding")
def business_data():
    return render_template('businessunderstanding.html')

@app.route("/dataunderstanding")
def data_understanding():
    return render_template('dataunderstanding.html')

@app.route("/dataengineering")
def data_engineering():
    return render_template('dataengineering.html')

@app.route("/crispmlmethodology")
def crisp_ml_methodology():
    return render_template('crispml.html')
if __name__ == "__main__":    app.run(debug=True)



