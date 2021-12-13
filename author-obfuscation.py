#encoding = utf-8
import os
import collections
import natsort
import re
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import contractions
import codecs
import json
import gensim

from logger import log

"""
Created on Mon Dec 30 05:43:00 2019

@author: fahrulrozim
"""


#Make data Truth json
'''Requirements :
    1. Data original
    2. Obfuscated Data
    3. Metadata (Start charpos, end charpos, id obfuscated)
Let's hope this will work out better'''

'''[ 
    { 
        "original": "The quick brown fox jumps over the lazy dog.", 
        "original-start-charpos": 10, 
        "original-end-charpos": 55, 
        "obfuscation": "Lazy lay the dog when an auburn fox quickly jumped over him.", 
        "obfuscation-id": 1 
    }, 
    { 
        "original": "Squdgy fez, blank jimp crwth vox!", 
        "original-start-charpos": 56, 
        "original-end-charpos": 70, 
        "obfuscation": "A short brimless felt hat barely blocks out the sound of a Celtic violin!", 
        "obfuscation-id": 2 
    }, 
    ... 
    ]'''

def load_vector(vector_data):
    model = gensim.models.KeyedVectors.load_word2vec_format(vector_data, binary=True)
    return model

def clean_document(document):
    clean = " ".join(document.splitlines())
    if '\ufeff' in clean:
        clean.replace('\ufeff', '')
    return clean

def replace_contractions(document):
    """Replace contractions in string of text"""
    return contractions.fix(document)

def sentence_tokenization(document):
    """Tokenize document to sentences"""
    return sent_tokenize(document)

def word_frequency(document):
    word_freq = []
    clear_document = clean_document(document)
    tokenized_document = tokenization(clear_document)
    for w in tokenized_document:
        word_freq.append(tokenized_document.count(w))
    return word_freq

def char_pos(document):
    sentences = document
    sent_list = []
    sent_length = []
    pos = []
    list_start = []
    list_end = []

    """separate each char"""
    for i, j in enumerate(sentences):
        list_obj = list(j)
        sent_list.append(list_obj)

    """Get each sentence length"""
    for i, j in enumerate(sent_list):
        if i==len(sent_list)-1:
            sent_length.append(len(j))
        else:
            sent_length.append(len(j)+1)

        if i==0:
            start = 0
            end = len(j)-1
        else:
            start = pos[i-1][1]+1
            end = start + len(j)
        pos.append((start, end))
        list_start.append(start)
        list_end.append(end)
    return list_start, list_end, pos, i, sentences

def build_truth_data(document, charpos, obfuscated):
    truth_dict = dict.fromkeys(['original', 'original-start-charpos', 'original-end-charpos', 'obfuscation',  'obfuscation-id'])
    truth_data = []
    id_obfucate = []
    original_list = document
    original_start_pos = charpos[0]
    original_end_pos = charpos[1]

    """Get obfucation id"""
    for i, j in enumerate(document):
        id_obfuscate.append(i)

    """Get obfucation data"""
    obfucated_data = obfuscated
    

    for i in range(len(id_obfuscate)):
        truth_dict['original'] = original_list[i]
        truth_dict['original-start-charpos'] = original_start_pos[i]
        truth_dict['original-end-charpos'] = original_end_pos[i]
        truth_dict['obfucation'] = obfuscated_data[i]
        truth_dict['obfuscation-id'] = id_obfucate[i]
        truth_data.append(truth_dict.copy())

    return truth_data

def tokenization(document):
    tokenized_sentence = []
    tokenizer = nltk.tokenize.TreebankWordTokenizer()
    for i in range(0, len(document)):
        tokenized_sentence.append(tokenizer.tokenize(document[i]))
    tokenized_document = word_tokenize(document)
    stopword = set(stopwords.words('english'))

    return tokenized_sentence

def POS_tag(document):
    pos_tag = []

    for i in range(0, len(document)):
        """Append tokenization and do pos tagging"""
        pos_tag.append(nltk.pos_tag(document[i]))

    return pos_tag

def check_stopwords(document):
    stopword = set(stopwords.words('english'))
    return stopword

def generate_synonym(document, pos_tag, model):
    flat_list = []
    detected_noun = []
    synonym_data = []
    word_list = []

    """Make the list flat first"""
    for i in range(0, len(document)):
        indexed_sentence = list(enumerate(pos_tag[i]))
        sentence_list = []
        for j in indexed_sentence:
            sentence_list.append(([j[0], j[1][0], j[1][1]]))
        flat_list.append(sentence_list)

    """Get set of noun and verb data"""
    for i in flat_list:
        temp = []
        for j in i:
            if j[2] in ['VB', 'VBG', 'VBD', 'VBP', 'VBZ', 'VBN', 'NN', 'NNP', 'NNS', 'NNPS']:
                temp.append(j)
        detected_noun.append(temp)

    """Generate synonym of detected words"""
    for x in detected_noun:
        temp = []
        for i, word, tag in x:
            if word not in model.wv.vocab:
                if '_' in word:
                    word = word.replace('_', ' ')
                temp.append((i, word, tag))
            else:
                word = model.wv.most_similar(word)[0][0]
                if '_' in word:
                    word = word.replace('_', ' ')
                temp.append((i, word, tag))

        synonym_data.append(temp)

    """Replace synonym to data"""
    replaced_words = flat_list[:]
    for i in range(len(synonym_data)):
        for j in synonym_data[i]:
            replaced_words[i][j[0]] = j
    
    """Get the words that replaced. This will make enumerate and pos tag discarded"""
    for i in replaced_words:
        temp = []
        for j in i:
            temp.append(j[1])
        word_list.append(temp)

 #   """Create the sentences from word lists"""
 #   word_lists = []
 #   for sublist in word_list:
 #       capitalized = False
 #       for val in sublist:
 #           if not capitalized and val[0].isalpha():
 #               word_lists.append(val.capitalize())
 #               capitalized = True
 #           else:
 #               word_lists.append(val)
 #   join_words = ' '.join(word_lists)

 #   """Normalize document"""
 #   new_document = re.sub(r'\s+(?=[^\w\d\s])', '', join_words)

    return word_list

def join_sentence(document):
    flat_words = []
    for sublist in document:
        capitalized = False
        for val in sublist:
            if not capitalized and val[0].isalpha():
                flat_words.append(val.capitalize())
                capitalized = True
            else:
                flat_words.append(val)
    generated_sentence = ' '.join(flat_words)
    return generated_sentence

def normalize_document(document):
    new_document = re.sub(r'\s+(?=[^\w\d\s])', '', document)
    return new_document


basepath = 'D:/Skripsi/data/pan16-author-masking-training-dataset-2016-02-17'
word2vec_data = 'D:/Skripsi/data/Google-pre-trained-data.bin'
def build_dataset(basepath, pre_trained):
    data = []
    problems = []
    same_authors = []
    truth_dict = collections.defaultdict(str)
    word2vec = load_vector(pre_trained)

    for (path, dirs, files) in os.walk(basepath):
        if files:
            if truth.txt in files:
                truth_dict = load_truth_dict(path):
            elif 'original.txt' in files:
                problem_name = os.path.basename(path)
                problems.append(problem_name)

                # untuk data same author
                known_files = [file for file in files if file.startswith('same-author') and file.endswith('.txt')]
                same_authors.append(known_files)

                #data original (obfuscated)
                unknown_files = [file for file in files if file.startswith('original') and file.endswith('.txt')]
                data.append(unknown_files)

                work_file = open(os.path.join(path, 'original.txt'), encoding='utf-8').read()

                """Try to iterate the files each time to go inside and process it"""
                get_charpos_original = char_pos(work_file)
                remove_lines = clean_document(work_file)
                contraction = replace_contractions(remove_lines)
                sentence = sentence_tokenization(contraction)
                token = tokenization(sentence)
                tagging = POS_tag(token)
                get_synonym = generate_synonym(sentence, tagging, word2vec)
                make_document = join_sentence(get_synonym)
                normalized_doc = normalize_document(make_document)

                """Pre process the obfuscated so we can make truth data"""
                processed_obf = sentence_tokenization(normalized_doc)
                charpos = char_pos(sentence)
                last_data = build_truth_data(sentence, charpos, processed_obf)

                """Save document for each times obfuscated"""
                f = open('obfuscated.txt', 'w')
                f.write(normalized_doc)
                f.close
                
    return data, problem, same_authors

build_dataset(basepath, word2vec_data)
