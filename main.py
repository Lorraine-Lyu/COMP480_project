import helper as hp
import pandas as pd
import csv
import random
import rambo as rb
from encoder import autoencoder



# data = pd.read_csv('user-ct-test-collection-01.txt', sep="\t")
# urllist = data.ClickURL.dropna().unique()
words = ["this", "is", "a", "test", "list", "for", "the", "autoencoder"]
vecs = [hp.word_vector_ascii(w) for w in words]

autoencoder.fit(vecs, vecs,
                epochs=10,
                shuffle=True)

autoencoder.call(vecs[0])