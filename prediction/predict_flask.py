import os
from flask import Flask, render_template, request, jsonify

from predict import PredictSentiment

app = Flask(__name__)


@app.route('/predict', methods=['GET', 'POST'])
def predict():

    content = request.json
    print(content)
    sentence = content['text']
    model_type = content['model']

    PD = PredictSentiment(model_type=model_type)
    preprocess_sentence = PD.preprocess_data(sentence)
    result = PD.predict_sentiment(preprocess_sentence)
    max_prediction=max(result)[1]

    return jsonify({"prediction_values":str(result), 'prediction':max_prediction})



@app.route('/predict_page/<sentence>/<model>', methods=['GET', 'POST'])
def predict_page(sentence, model):
    if request.method == 'GET':
        PD = PredictSentiment(model_type=model)
        preprocess_sentence = PD.preprocess_data(sentence)
        result = PD.predict_sentiment(preprocess_sentence)
        max_prediction=max(result)[1]

        return render_template('index.html', prediction=result, max_prediction=max_prediction)

if __name__ == '__main__':

    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)