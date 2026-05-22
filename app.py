from flask import Flask, render_template, request
import numpy as np
from models import rf_model, rf_encoder, rf_le_dept, rf_le_event

print("EVENT CLASSES:")
print(rf_le_event.classes_)

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

@app.route("/rforestconcept")
def rforest_concept():
    return render_template('Rforest/rforestconcepts.html')

@app.route("/rforestapplication", methods=['GET', 'POST'])
def rforest_application():
    prediction = None
    if request.method == 'POST':
        dept      = request.form['department']
        event     = request.form['event']
        month     = int(request.form['month'])
        year      = int(request.form['year'])
        count     = int(request.form['event_count'])
        diversity = int(request.form['event_diversity'])
        rainy     = 1 if month in [4, 5, 10, 11] else 0

        dept_enc  = rf_le_dept.transform([dept])[0]
        event_enc = rf_le_event.transform([event])[0]

        X = np.array([[count, diversity, rainy, year, month, dept_enc, event_enc]])
        pred      = rf_model.predict(X)[0]
        prediction = rf_encoder.inverse_transform([pred])[0]

    departments = sorted(rf_le_dept.classes_.tolist())
    events      = sorted(rf_le_event.classes_.tolist())
    return render_template('Rforest/rforestapplication.html',
                           prediction=prediction,
                           departments=departments,
                           events=events)

if __name__ == "__main__":    app.run(debug=True)



