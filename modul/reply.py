import json
import os
import pickle
import random
import re
import nltk

import numpy as np
import pandas as pd

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

from keras import Input, Model
from keras.activations import softmax
from keras.layers import Embedding, LSTM, Dense, Bidirectional, Concatenate
from keras_preprocessing.sequence import pad_sequences


def setConfig(file_name):
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    punct_re_escape = re.compile('[%s]' % re.escape('!"#$%&()*+,./:;<=>?@[\\]^_`{|}~'))
    unknowns = ["gak paham", "kurang ngerti", "I don't know"]
    path = file_name + "/"
    return factory, stemmer, punct_re_escape, unknowns, file_name, path


def load_config(path, config_path):
    data = {}
    if os.path.exists(path + config_path):
        with open(path + config_path) as json_file:
            data = json.load(json_file)
    return data


def load_tokenizer(path, tokenizer_path):
    tokenizer = None
    if os.path.exists(path + tokenizer_path):
        with open(path + tokenizer_path, 'rb') as handle:
            tokenizer = pickle.load(handle)
    if tokenizer:
        print(f"Tokenizer vocabulary size: {len(tokenizer.word_index) + 1}")  # +1 for padding token
    return tokenizer


def setParams(path, slang_path, config_path, tokenizer_path):
    list_indonesia_slang = pd.read_csv(path + slang_path, header=None).to_numpy()
    config = load_config(path, config_path)
    tokenizer = load_tokenizer(path, tokenizer_path)
    
    VOCAB_SIZE = len(tokenizer.word_index) + 1  # +1 for padding token
    maxlen_questions = config['maxlen_questions']
    maxlen_answers = config['maxlen_answers']
    return list_indonesia_slang, VOCAB_SIZE, maxlen_questions, maxlen_answers, tokenizer


def check_normal_word(word_input):
    slang_result = dynamic_switcher(data_slang, word_input)
    if slang_result:
        return slang_result
    return word_input


def normalize_sentence(sentence):
    sentence = punct_re_escape.sub('', sentence.lower())
    sentence = sentence.replace('iiteung', '').replace('\n', '')
    sentence = sentence.replace('iteung', '')
    sentence = sentence.replace('teung', '')
    sentence = re.sub(r'((wk)+(w?)+(k?)+)+', '', sentence)
    sentence = re.sub(r'((xi)+(x?)+(i?)+)+', '', sentence)
    sentence = re.sub(r'((h(a|i|e)h)((a|i|e)?)+(h?)+((a|i|e)?)+)+', '', sentence)
    sentence = ' '.join(sentence.split())
    if sentence:
        sentence = sentence.strip().split(" ")
        normal_sentence = " "
        for word in sentence:
            normalize_word = check_normal_word(word)
            root_sentence = stemmer.stem(normalize_word)
            normal_sentence += root_sentence + " "
        return punct_re_escape.sub('', normal_sentence)
    return sentence


def str_to_tokens(sentence, tokenizer, maxlen_questions):
    words = sentence.lower().split()
    tokens_list = list()
    for current_word in words:
        result = tokenizer.word_index.get(current_word, '')
        if result != '':
            tokens_list.append(result)
    return pad_sequences([tokens_list], maxlen=maxlen_questions, padding='post')


def setEncoderDecoder(VOCAB_SIZE):
    enc_inputs = Input(shape=(None,))
    enc_embedding = Embedding(VOCAB_SIZE, 256, mask_zero=True)(enc_inputs)
    _, forward_h, forward_c, backward_h, backward_c = Bidirectional(LSTM(256, return_state=True, dropout=0.5, recurrent_dropout=0.5))(enc_embedding)

    state_h = Concatenate()([forward_h, backward_h])
    state_c = Concatenate()([forward_c, backward_c])

    enc_states = [state_h, state_c]

    dec_inputs = Input(shape=(None,))
    dec_embedding = Embedding(VOCAB_SIZE, 256, mask_zero=True)(dec_inputs)
    dec_lstm = LSTM(256 * 2, return_state=True, return_sequences=True, dropout=0.5, recurrent_dropout=0.5)

    dec_outputs, _, _ = dec_lstm(dec_embedding, initial_state=enc_states)

    dec_dense = Dense(VOCAB_SIZE, activation=softmax)
    output = dec_dense(dec_outputs)

    return dec_lstm, dec_embedding, dec_dense, dec_inputs, enc_inputs, enc_states, output


def make_inference_models(dec_lstm, dec_embedding, dec_dense, dec_inputs, enc_inputs, enc_states):
    dec_state_input_h = Input(shape=(256 * 2,))
    dec_state_input_c = Input(shape=(256 * 2,))
    dec_states_inputs = [dec_state_input_h, dec_state_input_c]

    dec_outputs, state_h, state_c = dec_lstm(dec_embedding, initial_state=dec_states_inputs)
    dec_states = [state_h, state_c]

    dec_outputs = dec_dense(dec_outputs)

    dec_model = Model(inputs=[dec_inputs] + dec_states_inputs, outputs=[dec_outputs] + dec_states)
    enc_model = Model(inputs=enc_inputs, outputs=enc_states)

    return enc_model, dec_model


def setModel(enc_inputs, dec_inputs, output, dec_lstm, dec_embedding, dec_dense, enc_states, file_path):
    model = Model([enc_inputs, dec_inputs], output)
    model.load_weights(file_path)

    enc_model, dec_model = make_inference_models(dec_lstm, dec_embedding, dec_dense, dec_inputs, enc_inputs, enc_states)
    return model, enc_model, dec_model


def chat(input_value, tokenizer, maxlen_answers, enc_model, dec_model):
    input_value = stemmer.stem(
        normalize_sentence(normalize_sentence(input_value))
    )

    states_values = enc_model.predict(
        str_to_tokens(input_value, tokenizer, maxlen_questions)
    )

    empty_target_seq = np.zeros((1, 1))
    empty_target_seq[0, 0] = tokenizer.word_index['start']

    stop_condition = False
    decoded_translation = ''
    status = "false"

    kecocokan = 0

    while not stop_condition:
        dec_outputs , h , c = dec_model.predict([ empty_target_seq ] + states_values )
        sampled_word_index = np.argmax(dec_outputs[0, -1, :])
        kecocokan = dec_outputs[0, -1, sampled_word_index]
        if dec_outputs[0, -1, sampled_word_index] < 0.1:
            decoded_translation = unknowns[random.randint(0, (len(unknowns) - 1))]
            break
        sampled_word = None
        for word, index in tokenizer.word_index.items():
            if sampled_word_index == index:
                if word != 'end':
                    decoded_translation += ' {}'.format(word)
                sampled_word = word

        if sampled_word == 'end' or len(decoded_translation.split()) > maxlen_answers:
            stop_condition = True
            print(len(decoded_translation.split()));

        empty_target_seq = np.zeros((1, 1))
        empty_target_seq[0, 0] = sampled_word_index
        states_values = [h, c]
        status = "true"

    return decoded_translation.strip(), str(status).lower(), dec_outputs, kecocokan


factory, stemmer, punct_re_escape, unknowns, file_name, path = setConfig("output_dir_ws")

list_indonesia_slang, VOCAB_SIZE, maxlen_questions, maxlen_answers, tokenizer = setParams(path,
                                                                                          'daftar-slang-bahasa-indonesia.csv',
                                                                                          'config.json',
                                                                                          'tokenizer.pickle')

dec_lstm, dec_embedding, dec_dense, dec_inputs, enc_inputs, enc_states, output = setEncoderDecoder(VOCAB_SIZE)

model, enc_model, dec_model = setModel(enc_inputs,
                                       dec_inputs,
                                       output,
                                       dec_lstm,
                                       dec_embedding,
                                       dec_dense,
                                       enc_states,
                                       path + 'model-' + file_name + '.h5')

data_slang = {}
for key, value in list_indonesia_slang:
    data_slang[key] = value


def dynamic_switcher(dict_data, key):
    return dict_data.get(key, None)


def botReply(message):
    return chat(message, tokenizer, maxlen_answers, enc_model, dec_model)


def get_val_data(dir='output_dir_ws'):
    data = pd.read_csv(dir+'/val.csv')
    return data