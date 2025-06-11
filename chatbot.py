import tensorflow as tf
import numpy as np
import random
import json
import nltk
import pickle
from nltk.stem.lancaster import LancasterStemmer

nltk.download("punkt")
stemmer = LancasterStemmer()

with open(r"C:\Users\tejas\chatbot_files\intents.json") as json_data:
    intents = json.load(json_data)

data = pickle.load(open("training_data", "rb"))
words = data['words']
classes = data['classes']

model = tf.keras.models.load_model("model.h5")

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=False):
    sentence_words = clean_up_sentence(sentence)
    bag = [0]*len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print("Found in bag: %s" % w)
    return np.array(bag)

context = {}
ERROR_THRESHOLD = 0.25

def classify(sentence):
    input_data = np.expand_dims(bow(sentence, words), axis=0)
    results = model.predict(input_data)[0]
    results = [[i, r] for i, r in enumerate(results) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append((classes[r[0]], r[1]))
    return return_list

def response(sentence, userID='123', show_details=False):
    results = classify(sentence)
    if results:
        while results:
            for i in intents['intents']:
                if i['tag'] == results[0][0]:
                    if 'context_set' in i:
                        if show_details:
                            print('context:', i['context_set'])
                        context[userID] = i['context_set']
                    if not 'context_filter' in i or (userID in context and 'context_filter' in i and i['context_filter'] == context[userID]):
                        if show_details:
                            print('tag:', i['tag'])
                        return random.choice(i['responses'])
            results.pop(0)
    return "Sorry, I didn't understand that."

