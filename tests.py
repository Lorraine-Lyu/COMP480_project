import csv
import random

import pandas as pd

import helper as hp
import rambo as rb
from encoder import model
import trainer

import os


# step1: process data from the pdf directory
# all_pdfs = prepare_pdf_set()
# ")
# step2: initiate the autoencoder, train the model with
# the set of words extracted from all pdfs
samples = trainer.get_most_freq_words()
print("finished preparing test set")
trainer.train_model(model)
print("finished training autoencoder")
print("========testing fp rate of bloomfilter built with encoder===========")
for j in range(14, 28):
    print("test bit array size pow(2," , j, ")")
    trainer.test_collision_rate(model, samples, pow(2, j))

print("==========testing LSH property of hash function==========")

for i in range(12, 24):
    print("test LSH property pow(2,", i, ")")
    trainer.test_LSH_property(model, samples, pow(2, i))

#============= use the encoder as the hash function for LSH ==========

