import csv
import random

import pandas as pd

import helper as hp
import rambo as rb
from encoder import model
import trainer

import os

#pdf_names = ["Unit_5_Problem_Set.pdf"]


def prepare_pdf_set():
    rtn = set()
    for file in os.listdir("./pdf"):
        path = "pdf/" + file
        wrapper = hp.pdf_extract(path)
        rtn.add(wrapper)
    return rtn


def get_traing_set(pdf_sets):
    rtn = set()
    for pdf in pdf_sets:
        rtn = rtn.union(pdf.words)
    return rtn


def train_encoder(encoder, sets):
    training_set = list(get_traing_set(sets))
    print(training_set)
    encoder.fit(training_set, training_set, epochs=10, shuffle=True)


# step1: process data from the pdf directory
all_pdfs = prepare_pdf_set()
print("finished parsing pdf files")
# step2: initiate the autoencoder, train the model with
# the set of words extracted from all pdfs
trainer.train_model(model)
print("finished training autoencoder")
# step3: initiate rambo, use the trained autoencoder
# as the hash function.
k = 6
b = 4
r = 1000
# TODO: make sure encoder works as the hash function
#rb_table = rb.Rambo(k, b, r, [encoder])
#rb_table.insert_sets(all_pdfs)
#print("finished setting up rambo")
# accept user's input and query for pdf
#while True:
#    word = input("input a word:")
#    result = rb_table.query(word)
#    print(result)
