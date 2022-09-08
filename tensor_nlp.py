import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import numpy as np 
import re
import tweepy
import time
import random

CONSUMER_KEY = '4JPyNQXraM7R7TzV9EIXzX7nS'
CONSUMER_SECRET = 'gEBRrlSFdWrw1nWJW3spNf9efRfFMasVmrixBFx2XGB0WjoxT6'
ACCESS_KEY = '806001683490869248-bfM4Rg7wckJQcG92VPsqYtMWRNjOW24'
ACCESS_SECRET = 'SPeHC2r1f2VALDVGJF6WRes9ITBa08MqxMLBtIGAddIqN'

auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
auth.set_access_token(ACCESS_KEY, ACCESS_SECRET)
api = tweepy.API(auth)
# settings = api.search_tweets(q='sfchronicle', result_type = 'popular', count = 1, until = "2022-09-05")
# print(settings[0].text[:20])
home = api.home_timeline()
text = home[0].text
words = text.split()
tweet = []
for i in range(5):
    tweet.append(words[i])

tweet = ' '.join(tweet)

def train_ai():
    data = open('popular-poems.txt', encoding='utf8').read()
    data = data.strip()
    # Read, then decode for py2 compat.
    tokenizer = Tokenizer()

    # split each line into test chunks
    corpus = data.lower().split("\n")

    tokenizer.fit_on_texts(corpus)
    total_words = len(tokenizer.word_index) + 1

    input_sequences = []
    for line in corpus:
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i+1]
            input_sequences.append(n_gram_sequence)

    # pad sequences so each sequence length is the same amount
    # pads sequences based on the longest sequence
    max_sequence_len = max([len(x) for x in input_sequences])
    input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))


    # create predictors and label
    # slices first dimension then second, ":" means go through each item in first
    xs, labels = input_sequences[:,:-1],input_sequences[:,-1]

    ys = tf.keras.utils.to_categorical(labels, num_classes=total_words)


    # make the model that will figure out the next word
    model = Sequential()
    model.add(Embedding(total_words, 100, input_length=max_sequence_len-1))
    # keeps other words in memory so sentence makes more sense
    model.add(Bidirectional(LSTM(100)))
    model.add(Dense(total_words, activation='sigmoid'))
    adam = Adam(learning_rate=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

    history = model.fit(xs, ys, epochs=12, verbose=1)

    seed_text = tweet
    next_words = random.randint(25,30)
    
    for i in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        predicted = np.argmax(model.predict(token_list), axis=-1)
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        seed_text += " " + output_word
    output = re.sub(r'(\w+) \1', r'\1', seed_text, flags=re.IGNORECASE)


    def tweet_output(output):
        api.update_status(output)

    tweet_output(output)

while True:
    train_ai()
    time.sleep(5000)




    
