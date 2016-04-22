import random
import json
import math
import numpy
from nltk.tokenize import TreebankWordTokenizer
from collections import defaultdict


def compute_idf(inv_idx, n_docs, min_df=1, max_df_ratio=0.90):
    """ Compute term IDF values from the inverted index.

    Words that are too frequent or too infrequent get pruned.


    Arguments
    =========

    inv_idx: an inverted index as above

    n_docs: int,
        The number of documents.

    min_df: int,
        Minimum number of documents a term must occur in.
        Less frequent words get ignored.

    max_df_ratio: float,
        Maximum ratio of documents a term can occur in.
        More frequent words get ignored.

    Returns
    =======

    idf: dict
        For each term, the dict contains the idf value.

    """
    idf = {}
    for term in inv_idx:
        df = len(inv_idx[term])
        if df >= min_df and float(df/n_docs) <= max_df_ratio:
            idf[term] = math.log(float(n_docs/(1+df)))
    return idf


def compute_doc_norms(index, idf, n_docs):
    """ Precompute the euclidean norm of each document.

    Arguments
    =========

    index: the inverted index as above

    idf: dict,
        Precomputed idf values for the terms.

    n_docs: int,
        The total number of documents.

    Returns
    =======

    norms: np.array, size: n_docs
        norms[i] = the norm of document i.
    """
    norms = [0 for _ in range(n_docs)]
    for term in index:
        for i, tf in index[term]:
            norms[i] += math.pow((tf*idf[term]), 2)

    norm_array = numpy.array(norms)
    return numpy.sqrt(norm_array)


class QuoteFinder:
    def __init__(self):
        self.movies_to_quotes = read_file("jsons/movies_to_quotes.json")
        self.movies = read_file("jsons/movies.json")

        # Initialize query tokenizer
        self.tokenizer = TreebankWordTokenizer()

        # Create quotes to movies dictionary:
        self.quotes_to_movies = {}
        self.quotes = []
        for movie in self.movies_to_quotes:
            for quote in self.movies_to_quotes[movie]:
                self.quotes_to_movies[quote] = movie
                self.quotes.append(quote)
        # Create inverted index mapping terms to quotes
        self.inverted_index = {}
        for quote_id, quote in enumerate(self.quotes):
            quote_tokens = self.tokenizer.tokenize(quote)
            for term in quote_tokens:
                if term in self.inverted_index:
                    lst = self.inverted_index[term]
                    found = False
                    for i, tup in enumerate(lst):
                        if quote_id == tup[0]:
                            lst[i] = (quote_id, tup[1] + 1)
                            found = True
                    if not found:
                        self.inverted_index[term].append((quote_id, 1))
                else:
                    self.inverted_index[term] = [(quote_id, 1)]
        # Compute idf values for each term
        self.idf = compute_idf(self.inverted_index, len(self.quotes))
        # Prune out values removed by idf
        self.inverted_index = {key: val for key, val in self.inverted_index.items() if key in self.idf}
        # Compute document norms
        self.norms = compute_doc_norms(self.inverted_index, self.idf, len(self.quotes))

    def find_random(self):
        result = []
        r = random.randint(0, len(self.quotes))
        result.append(self.quotes[r] + " - \"" +self. movies[r] + "\"")
        return result

    def find_similar(self, query):
        query_words = self.tokenizer.tokenize(query)
        query_tfidf = defaultdict(int)
        for word in query_words:
            query_tfidf[word] += 1
        for word in query_tfidf:
            if word in self.idf:
                query_tfidf[word] *= self.idf[word]
            else:
                query_tfidf[word] = 0
        query_norm = 0
        for word in query_tfidf:
            query_norm += math.pow(query_tfidf[word], 2)
        query_norm = math.sqrt(query_norm)

        if query_norm == 0:
            return self.find_random()

        scores = [0 for _ in self.quotes]
        for word in query_tfidf:
            if word in self.inverted_index:
                for quote_id, tf in self.inverted_index[word]:
                    scores[quote_id] += query_tfidf[word]*tf*self.idf[word]

        results = []
        for i, s in enumerate(scores):
            if self.norms[i] != 0:
                results.append((s/(self.norms[i]*query_norm), i))
        results.sort(reverse=True)
        result_quotes = ["{} - \"{}\"".format(self.quotes[i], self.quotes_to_movies[self.quotes[i]]) for _, i in results]
        return result_quotes

    def find_final(self, q):
        return "placeholder response"


def read_file(path):
    # path = Docs.objects.get(id = n).address;
    # above line gives error "no such table: project_template_docs"
    file = open(path)
    transcripts = json.load(file)
    return transcripts
