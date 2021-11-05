from flask import Flask, render_template, request
import numpy as np
import pickle

model = pickle.load(open('kyphosis.pkl', 'rb'))


app = Flask(__name__)

@app.route('/')
def main():
    return render_template("home.html")

@app.route('/predict', methods=['POST'])
def home():
    data1 = request.form['a']
    data2 = request.form['b']
    data3 = request.form['c']

    arr = np.array([[data1, data2, data3]])
    pred = model.predict(arr)

    return render_template("after.html", data=pred)


if __name__ == '__main__':
    app.run(debug=True)