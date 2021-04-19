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
    def __init__(self, k, b, range, hash):
        self.k = k
        self.b = b
        self.range = range
        self.assignments = []
        self.tables = self.init_tables(k, b, range, hash)

    def init_tables(self, k, b, r, hash):
        rtn = []
        for i in range(0, k):
            row = []
            for j in range(0, b):
                cell = bf.BloomFilter()
                # the first input is the expected number 
                # of word to be inserted into each BF
                # will figure out later.
                cell.init_with_size(20, r, hash)
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
                    self[i][j].insertSet(s)
    
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
        result = []
        for r in self.tables:
            subset = self.query_row(r, elem)
            result.append(subset)
        if len(result) == 0:
            return None
        rtn = result[0]
        for i in range(1, len(result)):
            rtn = rtn.intersect(result[i])
        return rtn


    def query_row(self, row, elem):
        rtn = set()
        for j in range(0, self.b):
            if (self.tables[row][j].query(elem)):
                rtn = rtn.union(self.assignments[row][j])
        return rtn
