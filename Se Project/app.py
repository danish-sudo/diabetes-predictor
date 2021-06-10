from flask import Flask, request, render_template
import pickle
import pandas as pd

app = Flask(__name__)
with open('model.pickle', 'rb') as handle:
    model = pickle.load(handle)


@app.route('/')
def hello_world():
    return render_template('index.html')


@app.route('/predict')
def pred_diab():
    return render_template('predict.html')


@app.route('/predict_diabetes', methods=['POST', 'GET'])
def predict():
    text1 = request.form['1']
    text2 = request.form['2']
    text3 = request.form['3']
    text4 = request.form['4']
    text5 = request.form['5']
    text6 = request.form['6']
    text7 = request.form['7']
    text8 = request.form['8']

    row_df = pd.DataFrame([pd.Series([text1, text2, text3, text4, text5, text6, text7, text8])])
    print(row_df)
    prediction = model.predict_proba(row_df)
    output = '{0:.{1}f}'.format(prediction[0][1], 2)
    output = str(float(output) * 100) + '%'
    return render_template('result.html',
                           pred=f'Probability of being DIABETIC.\nis : {output}')


if __name__ == '__main__':
    app.run()
    Debug = True
