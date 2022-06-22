import requests

def send_request(sentence, model):
    """_Request to predict function to get prediction_
        text: Sentence for sentiment prediction
        model: Options [binary, multiclass]
    """

    try:
        res = requests.post('http://localhost:5000/predict', json={"text":sentence, "model":model})
        if res.ok:
            print(res.json())
    except TypeError:
        print('Check input type is string for sentence and model type')

if __name__ == '__main__':

    sentence = 'Bananas have a low tolerance for being walked over. They\'re dangerous'
    model = 'binary'

    send_request(sentence, model)
