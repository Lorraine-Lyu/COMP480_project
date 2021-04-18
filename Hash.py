from sklearn.utils import murmurhash3_32
import numpy as np
import random
import string

a = 2
b = 7
c = 13
d = 5
p = 1048573
seed = 17

def hashFunctionFactory(size, seed):
    def f(elem):
        return murmurhash3_32(np.int32(elem), seed=seed, positive=True) % size
    return f

def hashFunctionFactoryStr(size, seed):
    def f(elem):
        return murmurhash3_32(str.encode(elem), seed=seed, positive=True) % size
    return f

# functions for question 1
def two_univ_hash(elem):
    return ((a * elem + b) % p) % 1024

def three_univ_hash(elem):
    return ((a * pow(elem, 2) + b * elem + c) % p) % 1024

def four_univ_hash(elem):
    return ((a * pow(elem, 3) + b * pow(elem, 2) + c * elem + d) % p) % 1024

murmur = hashFunctionFactory(1024, seed)

# functions for question 4
chars = list(string.printable)
random.seed(seed)

def generate_member_set(low, high, size):
    member_set = set()
    random.seed(seed)
    while len(member_set) < size:
        member_set.add(random.randint(low, high))
    return member_set

def generate_test_set(member_set, seed):
    test_set = set()
    random.seed(seed)
    while len(test_set) < 1000:
        x = random.randint(10000,99999)
        if (x not in member_set):
            test_set.add(x)
    return test_set

def generate_random_string(length):
    rtn = []
    for i in range(0, length):
        index = random.randint(0, len(chars) - 1)
        rtn.append(chars[index])
    return ''.join(rtn)

def generate_random_string_set(size):
    rtn = set()
    random.seed(seed)
    for i in range(0, size):
        length = random.randint(8, 50)
        rtn.add(generate_random_string(length))
    return rtn

def test_false_positive_rate(bf, test_set):
    count = 0
    for t in test_set:
        if bf.query(t):
            count+=1
    return count/len(test_set)