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
print("test bit array size pow(2, 14)")
trainer.test_collision_rate(model, samples, pow(2, 14))
print("test bit array size pow(2, 18)")
trainer.test_collision_rate(model, samples, pow(2, 18))
print("test bit array size pow(2, 19)")
trainer.test_collision_rate(model, samples, pow(2, 20))
print("test bit array size pow(2, 21)")
trainer.test_collision_rate(model, samples, pow(2, 21))
print("test bit array size pow(2, 22)")
trainer.test_collision_rate(model, samples, pow(2, 22))
print("test bit array size pow(2, 23)")
trainer.test_collision_rate(model, samples, pow(2, 23))
print("test bit array size pow(2, 25)")
trainer.test_collision_rate(model, samples, pow(2, 25))

