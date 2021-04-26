import BloomFilter as bf
import helper as hp


class PDF_words_wrapper:
    def __init__(self, name, words):
        self.name = name
        self.words = words


class Rambo:
    # k is number of rows,
    # b is the number of cells in each row
    # range is the size of each bloom filter
    def __init__(self, k, b, range, encoder):
        self.k = k
        self.b = b
        self.range = range
        self.encoder = encoder
        self.assignments = []
        self.tables = self.init_tables(k, b, range)

    def init_tables(self, k, b, r):
        rtn = []
        for i in range(0, k):
            row = []
            for j in range(0, b):
                cell = bf.BloomFilter()
                # the first input is the expected number
                # of word to be inserted into each BF
                # will figure out later.
                cell.init(0.01, pow(2,15))
                cell.set_encoder(self.encoder)
                row.append(cell)
            rtn.append(row)
        return rtn

    # sets is a set of pdf_words_wrapper.
    # should only be called once after init.
    def insert_sets(self, sets):
        for i in range(0, self.k):
            nsets = hp.random_partition(self.b, sets)
            self.assignments.append(self.extract_assignment(nsets, i))
            for j in range(0, self.b):
                for s in nsets[j]:
                    self.tables[i][j].insertSet(s.words)

    def extract_assignment(self, partition, row_index):
        r_names = []
        for p in partition:
            cell_names = set()
            for wrapper in p:
                cell_names.add(wrapper.name)
            r_names.append(cell_names)
        return r_names

    # elem is a words
    def query(self, elem):
        print("searching for datasets containing word ", elem)
        result = []
        for r in range(len(self.tables)):
            subset = self.query_row(r, elem)
            result.append(subset)
        if len(result) == 0:
            return None
        rtn = result[0]
        for i in range(1, len(result)):
            rtn = rtn.intersection(result[i])
        return rtn

    def query_row(self, row, elem):
        rtn = set()
        for j in range(0, self.b):
            if self.tables[row][j].query_with_encoder(elem):
                print(self.assignments[row][j], "contains the word", elem)
                rtn = rtn.union(self.assignments[row][j])
        return rtn
