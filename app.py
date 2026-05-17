from flask import Flask, render_template, request

app = Flask(__name__)


#-------------------- HOME --------------------
@app.route("/")
def home():
    return render_template('home.html')
#-------------------- PAGES --------------------

@app.route("/businessdata")
def business_data():
    return render_template('businessdata.html')

@app.route("/dataengineering")
def data_engineering():
    return render_template('dataengineering.html')



#-------------------- RUN --------------------
if __name__ == "__main__":
    app.run(debug=True)