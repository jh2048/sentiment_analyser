from pandas import Series
from pickle import load
from keras.models import load_model

from tensorflow.keras.preprocessing.sequence import pad_sequences

from preprocessing import DimensionalityReduction, TextConversion

MODEL_FILE_PATH = 'models/'

class PredictSentiment:

    def __init__(self, model_type='binary') -> None:
        print(model_type)
        
        if model_type == 'binary':
            MODEL_FILE_NAME = 'model_binary'
        else:
            MODEL_FILE_NAME = 'model_multiclass'

        print(MODEL_FILE_NAME)
        self.saved_model = load_model(f'{MODEL_FILE_PATH+MODEL_FILE_NAME}.h5')

        with open(f'{MODEL_FILE_PATH+MODEL_FILE_NAME}.label_encoder.pkl', "rb") as le_file:
            self.le = load(le_file)
            le_file.close()
        
        with open(f'{MODEL_FILE_PATH+MODEL_FILE_NAME}.tokenizer.pkl', "rb") as tknsr_file:
            self.tknsr = load(tknsr_file)
            tknsr_file.close()



    def preprocess_data(self, tweet_texts:list) -> Series:
        """Process data for prediction

        Args:
            tweet_texts (Series): Series of strings containg Text/ Tweets

        Returns:
            Series: Processed Texts/Tweets
        """

        dim_red = DimensionalityReduction()
        txt_con = TextConversion()

        tweet_texts = Series(tweet_texts)
        ## Initialise list for clean tweets
        tweet_texts = tweet_texts.apply(dim_red.remove_standardised_references)
        tweet_texts = tweet_texts.apply(txt_con.convert_sentence_to_lowercase)
        tweet_texts = tweet_texts.apply(txt_con.tokenise_text)
        tweet_texts = tweet_texts.apply(txt_con.text_lemmatisation)
        tweet_texts = tweet_texts.apply(dim_red.remove_stopwords)
        tweet_texts = tweet_texts.apply(dim_red.remove_punctuation)
        tweet_texts = tweet_texts.apply(lambda x: ' '.join(w for w in x))
        return tweet_texts


    def predict_sentiment(self, sentence:Series):

        X_test_seq = self.tknsr.texts_to_sequences(sentence)
        X_test_seq_pad = pad_sequences(X_test_seq, padding='post', maxlen=50)

        y_hat = self.saved_model.predict(X_test_seq_pad)

        return list(zip(y_hat[0], self.le.classes_))


if __name__ == '__main__':

    sentence = 'Bananas have a low tolerance for being walked over. They\'re dangerous'
    model = 'binary'

    PD = PredictSentiment(model_type=model)
    preprocess_sentence = PD.preprocess_data([sentence])
    result = PD.predict_sentiment(preprocess_sentence)
    max_prediction=max(result)[1]
    print(result, max_prediction)