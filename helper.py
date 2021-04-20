import random

from tika import parser

from rambo import PDF_words_wrapper


# Randomly partition a list into
# a list of k lists of elements.
def random_partition(k, lst):
    results = [[] for i in range(k)]
    for value in lst:
        x = random.randrange(k)
        results[x].append(value)
    return results


# name is the path to the pdf,
# extracts the text from the pdf
def pdf_extract(name):
    # creating a pdf file object
    raw = parser.from_file(name)
    content = raw["content"]
    content = set(content.split())
    return PDF_words_wrapper(name, content)


# converts a word into an int vector using ascii
def word_vector_ascii(word):
    v = []
    for i in range(0, 32):
        if i < len(word):
            v.append(ord(word[i]) / 128)
        else:
            v.append(0)
    return v
