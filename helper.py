import random

from tika import parser

from rambo import PDF_words_wrapper

# only gets three-gram of string
def three_gram(s):
    output = set()
    if (len(s) <= 3):
        output.add(s)
        return output
    for i in range(len(s)-2):
        output.add(s[i:i+3])
    return output

def Jaccard(s1, s2):
    if len(s1) == 0 or len(s2) == 0:
        return 0
    return len(s1.intersection(s2)) / len(s1.union(s2))

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
    print("parsed ", name, "total words ", len(content))
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
