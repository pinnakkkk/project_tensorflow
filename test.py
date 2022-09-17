from crypt import methods
from tkinter.messagebox import RETRY
from flask import Flask, render_template, request
import tensorflow as tf
from keras.datasets import imdb
from keras.utils import pad_sequences
import keras
import tensorflow as tf
import os
import numpy as np

app = Flask(__name__)

@app.route('/')
def index():
   return render_template("index.html")    

@app.route('/text', methods=['GET','POST'])
def main():
    MAXLEN = 250

    model = tf.keras.models.load_model('/home/lazylinuxer/tensorflow_projects/project_tensor/rnn_text_review.h5')
    word_index = imdb.get_word_index()

    def encode_text(text):
        tokens = keras.preprocessing.text.text_to_word_sequence(text)
        tokens = [word_index[word] if word in word_index else 0 for word in tokens]
        return keras.utils.pad_sequences([tokens], MAXLEN)[0]

    def predict(text):
        encoded_text = encode_text(text)
        pred = np.zeros((1,250))
        pred[0] = encoded_text
        result = model.predict(pred)
        return result

    text = request.form.get('text_input')

    score = predict(text)
        # if predict(text) > 50:
        #     print("(score):",predict(text)," Positive")

    return render_template('index.html', comments=score)

if __name__ == '__main__':
    app.run(port=3000, debug=True)
