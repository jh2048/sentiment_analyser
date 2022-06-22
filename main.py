import os
import pandas as pd
from pandas import Series, DataFrame

from prediction.preprocessing import DimensionalityReduction, TextConversion
from training import CreateDatasets, ModelBuilder, Validation
from prediction.predict import PredictSentiment



def preprocess_data(tweet_texts:Series, predict=False) -> Series:
    """Process data for training model

    Args:
        tweet_texts (Series): Series of strings containg Text/ Tweets
        predict (bool, optional): Apply methods only available during training? Defaults to False.

    Returns:
        Series: Processed Texts/Tweets
    """

    dim_red = DimensionalityReduction()
    txt_con = TextConversion()

    ## Initialise list for clean tweets
    tweet_texts = tweet_texts.apply(dim_red.remove_standardised_references)
    tweet_texts = tweet_texts.apply(txt_con.convert_sentence_to_lowercase)
    tweet_texts = tweet_texts.apply(txt_con.tokenise_text)
    tweet_texts = tweet_texts.apply(txt_con.text_lemmatisation)
    tweet_texts = tweet_texts.apply(dim_red.remove_stopwords)
    tweet_texts = tweet_texts.apply(dim_red.remove_punctuation)

    if not predict:
        dim_red.generate_single_use_word_list(tweet_texts)
        tweet_texts = tweet_texts.apply(dim_red.remove_single_use_words)

    tweet_texts = tweet_texts.apply(lambda x: ' '.join(w for w in x))
    return tweet_texts




def train_validate_model(df:DataFrame, model_type='binary') -> None:
    """Driver for building, running and validating model.

    Args:
        df (DataFrame): Dataframe containing all variables
    """

    CD = CreateDatasets()
    MB = ModelBuilder(model_type=model_type)
    
    text_col_name = 'tweet_text'
    emotion_col_name = 'is_there_an_emotion_directed_at_a_brand_or_product'
    strat_col_name = 'emotion_in_tweet_is_directed_at'

    ## Split dataset train/ test
    CD.create_train_test_set(dataframe=df, text_col_name=text_col_name, emotion_col_name=emotion_col_name, strat_col_name=strat_col_name)
    
    ## Preprocess and oversample training and test set
    X_train = preprocess_data(CD.X_train_over)
    X_test = preprocess_data(CD.X_test, predict=True)

    ## Convert texts to sequence of ints
    MB.convert_words_to_ints(X_train, X_test)

    ## Encode labels
    MB.encode_labels(CD.y_train_over, CD.y_test)

    ## Split dataset train/val
    CD.create_train_val_set(MB.X_train_seq_pad, MB.y_train_oh)

    ## Set up and train model
    MB.model_setup()
    results = MB.run_model(CD.X_train_emb, CD.y_train_emb, CD.X_val_emb, CD.y_val_emb)

    ## Validate model
    VD = Validation()
    VD.visualize_training_results(results)
    report = VD.validation(MB.X_test_seq_pad, MB.y_test_oh, MB.y_test_enc)

    print(report)


def predict(sentences:list, model_type:str='binary') -> dict:
    """Take in list of sentences and predict sentiment

    Args:
        sentences (list): list of strings
        binary_model (str, optional): Whether to use binary or multiclass model (True=binary, False=Multiclass). Defaults to 'binary'.

    Returns:
        dict: dict of predictions containing values generated and outcome prediction (based on max value)
    """

    PD = PredictSentiment(model_type=model_type)
    predictions = {}
    for idx, sentence in enumerate(sentences):
        result = PD.predict_sentiment([sentence])

        predictions[idx] = {}
        predictions[idx]['result'] = result
        predictions[idx]['max_prediction'] = max(result)[1]
        predictions[idx]['sentence'] = sentence
    
    return predictions
    



if __name__ == '__main__':

    
    DATA_FILEPATH = 'data/product_sentiment.csv'
    model_type = 'multiclass'

    df = pd.read_csv(DATA_FILEPATH)

    ## Remove unknown sentiments
    df_unk_removed = df[df['is_there_an_emotion_directed_at_a_brand_or_product'] != "I can't tell"]

    ## Polarity analysis?
    if model_type == 'binary':
        df_unk_removed = df_unk_removed[df_unk_removed['is_there_an_emotion_directed_at_a_brand_or_product'] != "No emotion toward brand or product"]

    ## Remove missing tweets
    df_dropna_tweets = df_unk_removed.dropna(subset=['tweet_text'])

    ## Convert NANs to UNK
    df_dropna_tweets.loc[:,'emotion_in_tweet_is_directed_at'] = df_dropna_tweets['emotion_in_tweet_is_directed_at'].apply(lambda x: 'UNK' if x != x else x)

    ## Drop NANs
    df_input = df_dropna_tweets.dropna(subset=['tweet_text', 'is_there_an_emotion_directed_at_a_brand_or_product', 'emotion_in_tweet_is_directed_at'])

    ## Build, train and validate model
    train_validate_model(df_input, model_type=model_type)


    ## PREDICTION
    sentences = [
        '#mobilephotography #iphone Shot and edited with IPhone_ XS max 📸✨📲',
        'Crazy how fast something can grow in 11 years! 🍎 #iphone #iOS16',
        'Well, here is an interesting glitch. 😂 Look at my time, WiFi, battery and cellular indicators. 🤣. That ain’t right.#Apple #iPhone #iOS',
        'Completely agree. My #iphone XR (3 years old) now takes 2 to 3 times longer to open basic apps such as the camera for no apparent reason.',
        'This is real sh*t, it’s been 8 hrs since i got this notification.(phone is waterproof though) And I can\'t charge my phone. And battery is draining like fountain even though i\'m not using it. Really frustated!!!' 
    ]

    result = predict(sentences, model_type=model_type)
    print(result)


    ## Implicit vs explicit sentiment
    ## Objective vs Subjective sentiment
    ## Bigrams, trigrams (Back off method)