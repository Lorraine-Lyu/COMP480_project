from collections import defaultdict

import tensorflow as tf

import trainer


def map_code_to_hash_val(code, r):
    # code is the output of autoencoder of a word
    # range is the length of the bloomfilter
    # returns an array of position indexes to be set to '1' in bloomfilter
    rtn = []
    # convert tensor to array
    arr = code.numpy()[0]
    #     print(arr)
    sub_range_len = r / 32
    # not sure about this value, should be the largest value of a tensor slot
    # in the encoded array
    max_val = 100
    for i in range(0, 32):
        if arr[i] > 0:
            pos = arr[i] * sub_range_len / max_val + i * sub_range_len
            #             pos = pos % sub_range_len
            rtn.append(int(pos) % r)
    return rtn


class Minhash:
    def __init__(self, encoder, size):
        self.table = defaultdict(set)
        self.encoder = encoder
        self.size = size

    def insert(self, elem):
        code = self.encoder.encoder(
            tf.convert_to_tensor([(trainer.word_to_ascii(elem))])
        )
        positions = map_code_to_hash_val(code, self.size)
        self.table[tuple(positions)].add(elem)

    def query(self, elem):
        code = self.encoder.encode(
            tf.convert_to_tensor([(trainer.word_to_ascii(elem))])
        )
        positions = map_code_to_hash_val(code, self.size)
        if elem in self.table[tuple(positions)]:
            return True
        else:
            return False

    def get_similar_elements(self, elem):
        code = self.encoder.encoder(
            tf.convert_to_tensor([(trainer.word_to_ascii(elem))])
        )
        positions = map_code_to_hash_val(code, self.size)
        return self.table[tuple(positions)]
