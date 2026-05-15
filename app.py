from flask import Flask, render_template, request

app = Flask(__name__)


#-------------------- HOME --------------------
@app.route("/")
def home():
    return render_template('1stphase/businessdata.html')

#-------------------- RUN --------------------
if __name__ == "__main__":
    app.run(debug=True)