import sys, os
import pandas as pd
import numpy as np
from pandas import Series, DataFrame
from numpy import argmax
from pickle import dump, load
import matplotlib.pyplot as plt
from imblearn.over_sampling import RandomOverSampler 

## sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

## keras
from keras import models, layers, metrics
from keras.models import load_model
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras.utils.np_utils import to_categorical

## tensorflow
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


## STATIC - ISH VALUES
RANDOM_STATE = 42
MODEL_FILE_PATH = '../prediction/models/'
MODEL_FILE_NAME = 'model_multiclass'
RUN_MODEL = True

class CreateDatasets:

    def __init__(self) -> None:
        pass


    def create_train_test_set(self, dataframe:DataFrame, text_col_name:str, emotion_col_name:str, strat_col_name:str, sampling:str='over') -> None:
        """Create class objects for train / test sets, split into variables (X) and response (y)

        Args:
            dataframe (DataFrame): Dataframe containing variables for train/test split
            text_col_name (str): Column name for column containing text data
            emotion_col_name (str): Column name for column containing response variable (sentiment)
            strat_col_name (str): Column name for column containing groups for stratifying
            sampling (str, optional): TBC - Add undersampling option. Defaults to 'over'.
        """

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(dataframe[text_col_name], 
                                                        dataframe[emotion_col_name],
                                                    test_size = 0.2,
                                                    random_state = RANDOM_STATE,
                                                    stratify = dataframe[strat_col_name])

        if sampling == 'over':
            self.X_train_over, self.y_train_over = self.dataset_oversampling(self.X_train, self.y_train)



    def create_train_val_set(self, X_train_seq_pad:np.array, y_train_oh:np.matrix) -> None:
        """Create class objects for train / validation sets

        Args:
            X_train_seq_pad (np.array): Numpy array with shape (len(sequences), maxlen (Output of pad_sequences)
            y_train_oh (np.matrix): A binary matrix representation of the input. (Output of to_categorical)
        """

        self.X_train_emb, self.X_val_emb, self.y_train_emb, self.y_val_emb = train_test_split(X_train_seq_pad, 
                                                                y_train_oh, 
                                                                test_size=0.1, 
                                                                random_state=RANDOM_STATE)

    def dataset_oversampling(self, X:Series, y:Series) -> pd.DataFrame:
        """Method used in create_train_test_set. Oversampling training data.

        Args:
            X (Series): X_train data
            y (Series): y_train data

        Returns:
            pd.DataFrame: Oversampled variables
            pd.Series: Oversample response
        """

        oversample = RandomOverSampler(sampling_strategy='all')
        X_over, y_over = oversample.fit_resample(DataFrame(X), y)
        print(y_over.value_counts())
        return X_over.squeeze(), y_over

class ModelBuilder:


    def __init__(self, max_tokens=50, num_words=5000, model_type='binary') -> None:
        self.MAX_TOKENS = max_tokens
        self.NUM_WORDS = num_words
        self.num_classes = 0
        self.model_type = model_type

        self.le = LabelEncoder()



    def encode_labels(self, y_train:Series, y_test:Series) -> None:
        """Encode response variables for use in modelling and converts to binary class matrix.
            Saves label encoder for later use.

        Args:
            y_train (Series): Response training data
            y_test (Series): Response test data
        """

        self.num_classes = len(y_train.value_counts())

        y_train_enc = self.le.fit_transform(y_train)
        self.y_test_enc = self.le.transform(y_test)
        self.y_train_oh = to_categorical(y_train_enc)
        self.y_test_oh = to_categorical(self.y_test_enc)

        with open(f'{MODEL_FILE_PATH+MODEL_FILE_NAME}.label_encoder.pkl', 'wb') as le_file:
            dump(self.le, le_file)
            le_file.close()



    def convert_words_to_ints(self, X_train:Series, X_test:Series) -> None:
        """Splits sentence into tokens and converts into sequence of integers. 
            Padded to ensure all sentences are the same length.
            Tokenizer saved for later use.

        Args:
            X_train (Series): Training set
            X_test (Series): Test set
        """

        tokenizer = Tokenizer(num_words=self.NUM_WORDS, filters='', lower=False, split=' ')
        tokenizer.fit_on_texts(X_train)

        X_train_seq = tokenizer.texts_to_sequences(X_train)
        X_test_seq = tokenizer.texts_to_sequences(X_test)

        self.X_train_seq_pad = pad_sequences(X_train_seq, padding='post', maxlen=self.MAX_TOKENS)
        self.X_test_seq_pad = pad_sequences(X_test_seq, padding='post', maxlen=self.MAX_TOKENS)

        with open(f'{MODEL_FILE_PATH+MODEL_FILE_NAME}.tokenizer.pkl', 'wb') as tknsr_file:
            dump(tokenizer, tknsr_file)
            tknsr_file.close()
    


    def model_setup(self, output_dim:int=128) -> None:
        """Sets up model for sentiment analysis. Includes model checkpoints.

        Args:
            output_dim (int, optional): Dimension of the embedding (z). Defaults to 128.
        """

        ## Automatically creates directory if not exists
        self.emb_model = models.Sequential()
        self.emb_model.add(layers.Embedding(input_dim=self.NUM_WORDS, output_dim=output_dim, input_length=self.MAX_TOKENS))
        self.emb_model.add(layers.Flatten())
        self.emb_model.add(layers.Dense(self.num_classes, activation='softmax'))
        self.early_stopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)


        self.save_best = ModelCheckpoint(os.path.join(MODEL_FILE_PATH,f'{MODEL_FILE_NAME}.h5'), 
                    monitor='val_categorical_accuracy', 
                    mode='max', 
                    verbose=1, 
                    save_best_only=True)

        if self.model_type == 'binary': loss_function = 'binary_crossentropy'
        else: loss_function = 'categorical_crossentropy'

        self.emb_model.compile(loss=loss_function, 
                optimizer='nadam', 
                metrics=[metrics.categorical_accuracy])



    def run_model(self, X_train_emb:np.array, y_train_emb:np.array, X_val_emb:np.array, y_val_emb:np.array, epochs:int=30) -> object:
        """ Run model with parameters previously defined

        Args:
            X_train_emb (np.array): Training variables
            y_train_emb (np.array): Training response
            X_val_emb (np.array): Validation variables
            y_val_emb (np.array): Validation response

        Returns:
            object: Record of training loss values and metrics values at successive epochs, as well as validation loss values and validation metrics values
        """

        results = self.emb_model.fit(X_train_emb, 
                                    y_train_emb, 
                                    validation_data=(X_val_emb, y_val_emb), 
                                    epochs=epochs,
                                    callbacks=[self.early_stopping, self.save_best])

        return results



class Validation:

    def __init__(self) -> None:
        self.saved_model = load_model(f'{MODEL_FILE_PATH+MODEL_FILE_NAME}.h5')
        with open(f'{MODEL_FILE_PATH+MODEL_FILE_NAME}.label_encoder.pkl', "rb") as le_file:
            self.le = load(le_file)
            le_file.close()

    def visualize_training_results(self, results:object, show_plot=False) -> None:
        """Visualising loss and accuracy. Outputs plots (PNG).

        Args:
            results (object): Results from sentiment model
            show_plot (bool): If show plot on screen (for jupyter notebook)
        """
        history = results.history
        plt.figure()
        plt.plot(history['val_loss'])
        plt.plot(history['loss'])
        plt.legend(['val_loss', 'loss'])
        plt.title('Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.savefig(f'../results/{MODEL_FILE_NAME}.training_results_LOSS.png')
        
        plt.figure()
        plt.plot(history['val_categorical_accuracy'])
        plt.plot(history['categorical_accuracy'])
        plt.legend(['val_categorical_accuracy', 'categorical_accuracy'])
        plt.title('Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.savefig(f'../results/{MODEL_FILE_NAME}.training_results_ACCURACY.png')


    def validation(self, X_test_seq_pad:np.array, y_test_oh:np.array, y_test_enc:np.array) -> None:
        """Validate model using test data.

        Args:
            X_test_seq_pad (np.array): Test set variables in format - sequence of integers. 
            y_test_oh (np.array): Test set response variables in format - binary class matrix
            y_test_enc (np.array): Response variable in encoded format
        """
        
        y_hat = self.saved_model.predict(X_test_seq_pad)
        self.results = self.saved_model.evaluate(X_test_seq_pad, y_test_oh)
        report = classification_report(y_test_enc, argmax(y_hat, axis=1), target_names=self.le.classes_)
        with open(f'../results/{MODEL_FILE_NAME}_classification_report.txt', 'w') as f:
            f.write(report)
            f.close()

        return report


