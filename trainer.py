import urllib.request
from pathlib import Path
import numpy as np
import pandas as pd
import re
from sklearn.model_selection import train_test_split
import tensorflow as tf

#================ This section of code preprocesses the training data ===================

AOL_URL = "http://www.cim.mcgill.ca/~dudek/206/Logs/AOL-user-ct-collection/user-ct-test-collection-01.txt"

data_dir = Path("data")
data_file = Path("data/aol.txt")
PAD_CONST = 512

# breaks phrases into word and collect unique ones.
def get_words(phrases):
    keywords = set()
    for s in phrases:
        for w in re.findall(r'\w+', s) :
            keywords.add(w)
    return list(keywords)

def word_to_ascii(word):
    ascii_word = list(map(ord, word))
    padded_ascii = ascii_word + ([0] * (PAD_CONST - len(ascii_word)))
    return padded_ascii

def preprocess_words():
    if not data_file.is_file():
        if not data_dir.is_dir():
            data_dir.mkdir(parents=True, exist_ok=True)

        with urllib.request.urlopen(AOL_URL) as data_url, data_file.open(
            "w", encoding="utf-8"
        ) as fd:
            fd.write(data_url.read().decode("utf-8"))

    data = pd.read_csv(data_file, sep="\t")
    phrases = data.Query.dropna().unique().tolist()

    word_set = get_words(phrases)

    phrases_ascii = np.array(list(map(word_to_ascii, word_set)))
    return phrases_ascii

#=================== The block below sets up the traning sets ========================

def setup_training_set(phrases_ascii):
    X_train, X_test, y_train, y_test = train_test_split(
        phrases_ascii, phrases_ascii, test_size=0.2, random_state=2021
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.25, random_state=2021
    )

    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))

    BATCH_SIZE = 32
    SHUFFLE_BUFFER_SIZE = 96

    train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
    val_dataset = val_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
    test_dataset = test_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)

    return (train_dataset, val_dataset, test_dataset)

#=============The section below trains the autoencoder========================
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", mode="min", patience=10, verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        factor=0.1, patience=5, min_lr=0.00001, verbose=1
    ),
    tf.keras.callbacks.ModelCheckpoint(
        "model-tgs-salt.h5", verbose=1, save_best_only=True, save_weights_only=True
    ),
]

def fit_model(train_dataset, model, val_dataset):
    history = model.fit(
        train_dataset, epochs=30, callbacks=callbacks, validation_data=val_dataset
    )

def train_model(model):
    ascii_list = preprocess_words()
    print("converted all training words to ascii vectors")
    (train_dataset, val_dataset, test_dataset) = setup_training_set(ascii_list)
    print("setted up all datasets for training")
    fit_model(train_dataset, model, val_dataset)