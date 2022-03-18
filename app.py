import pickle

from flask import Flask, render_template, request
from flask_cors import CORS, cross_origin

app = Flask(__name__)

scaler = pickle.load(open("scaler.pkl", "rb"))
model = pickle.load(open("xgb.pkl", "rb"))


@app.route('/', methods=['GET', 'POST'])
@cross_origin()
def Home():
    return render_template('index.html')


@app.route("/predict", methods=['GET', 'POST'])
@cross_origin()
def predict():
    if request.method == "POST":
        try:
            data_req = dict(request.form)
            data = data_req.values()
            data = [list(map(float, data))]
            print(data)
            scaled = scaler.transform(data)
            result = model.predict(scaled)
            if result[0] == 0:
                result = "<=50K"
            else:
                result = '>50K'
            return render_template("result.html", result=result)
        except Exception as e:
            error = {'error': e}
            return render_template('404.html', error=error)


if __name__ == "__main__":
    app.run(debug=True)
