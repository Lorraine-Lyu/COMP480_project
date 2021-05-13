import random
from math import log

import numpy as np

import bit
import Hash
import trainer
import tensorflow as tf

seed = 13

def map_code_to_hash_val(code, r):
    # code is the output of autoencoder of a word
    # range is the length of the bloomfilter
    # returns an array of position indexes to be set to '1' in bloomfilter
    rtn = []
    # convert tensor to array
    arr = code.numpy()[0]
#     print(arr)
    sub_range_len = r/32
    # not sure about this value, should be the largest value of a tensor slot
    # in the encoded array
    max_val = 100
    for i in range(0, 32):
        if arr[i] > 0:
            pos = arr[i] * sub_range_len / max_val + i * sub_range_len
#             pos = pos % sub_range_len
            rtn.append(int(pos) % r)
    return rtn

class BloomFilter:

    # python doesn't support multiple constructor
    # so first init an empty instance and then
    # call init functions respectively
    def __init__(self):
        self.expectedNumber = 0
        self.size = 0
        self.bf = None

    # the primary init function
    # size is calculated based on
    # expected number of values to be inserted
    def init(self, fp, num):
        self.expectedNumber = num
        self.computeSize(fp)
        self.bf = bit.makeBitArray(self.size)
        self.getHashFunctions()

    ## init with preset size
    ## used for q4.2
    # directly pass in a list of hash function
    # This was modified for project
    def init_with_size(self, num, size, hash=[]):
        self.expectedNumber = num
        self.size = size
        self.bf = bit.makeBitArray(self.size)
        self.hash = hash

    def computeSize(self, fp):
        self.size = round(self.expectedNumber * log(fp, 0.618))

    def set_encoder(self, encoder):
        self.encoder = encoder

    def getHashFunctions(self):
        functions = set()
        random.seed(seed)
        numHashFunction = round(self.size / self.expectedNumber * log(2, 10))
        members = set()
        for i in range(0, numHashFunction):
            members.add(random.randint(1, 1000))
        for m in members:
            f = Hash.hashFunctionFactory(self.size, m)
            functions.add(f)
        self.hash = functions

    def getHashFunctionsStr(self):
        functions = set()
        random.seed(seed)
        members = set()
        numHashFunction = round(self.size * 0.7 / self.expectedNumber)
        for i in range(0, numHashFunction):
            members.add(random.randint(1, 1000))
        for m in members:
            f = Hash.hashFunctionFactoryStr(self.size, m)
            functions.add(f)
        self.hash = functions

    def insert(self, elem):
        for h in self.hash:
            position = h(elem)
            bit.setBit(self.bf, position)


    def insert_with_encoder(self, elem):
        code = self.encoder.encoder(tf.convert_to_tensor([(trainer.word_to_ascii(elem))]))
        positions = map_code_to_hash_val(code, self.size)
        for pos in positions:
            bit.setBit(self.bf, pos)

    def query_with_encoder(self, elem):
        code = self.encoder.encoder(tf.convert_to_tensor([(trainer.word_to_ascii(elem))]))
        positions = map_code_to_hash_val(code, self.size)
        for pos in positions:
            if not bit.testBit(self.bf, pos):
                return False
        return True

    def insertSet(self, source):
        if not self.encoder:
            print("ERROR: insertSet is only applicable for BF with encoder")
        for s in source:
            self.insert_with_encoder(s)

    def query(self, elem):
        for h in self.hash:
            position = h(elem)
            if not bit.testBit(self.bf, position):
                return False
        return True

    def getBloomFilter(self):
        return self.bf
