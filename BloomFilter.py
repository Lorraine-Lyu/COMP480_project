import bit
from math import log
import numpy as np
import Hash
import random

seed = 13

class BloomFilter():

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
    def init_with_size(self, num, size, hash):
        self.expectedNumber = num
        self.size = size
        self.bf = bit.makeBitArray(self.size)
        self.hash = hash 

    def computeSize(self, fp):
        self.size = round(self.expectedNumber * log(fp, 0.618))

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
        numHashFunction = round(self.size * 0.7/self.expectedNumber)
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

    def insertSet(self, source):
        for s in source:
            self.insert(s)

    def query(self, elem):
        for h in self.hash:
            position = h(elem)
            if not bit.testBit(self.bf, position):
                return False
        return True

    def getBloomFilter(self):
        return self.bf

