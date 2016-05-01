import random
import json
import math
import numpy
import os.path
import io
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import TreebankWordTokenizer
from collections import defaultdict


def read_file(path):
    # path = Docs.objects.get(id = n).address;
    # above line gives error "no such table: project_template_docs"
    file = open(path)
    transcripts = json.load(file)
    return transcripts


def punct_strip(s):
    """Strips punctuation from a string and returns the lowercase version. Doesn't work for crazy punctuation

    Arguments
    =========

    s : a string

    Returns
    =======

    new_quote : a punctuation stripped string
    """
    punctuation = (".", "/", ":", ",", ";", "!", "?", "(", ")", '"', "-", "]", "'", "[", "#", "$")
    l = s.lower().split()
    new_quote = ""
    for x in l:
        if (x == "-"):
            continue
        else:
            new_x = x.strip()
            if (new_x.endswith("...")):
                new_x = new_x[:len(x) - 3]
            if new_x.endswith(punctuation):
                new_x = new_x[:len(x) - 1]
            newer_x = new_x
            if newer_x.startswith(punctuation):
                newer_x = newer_x[1:]
            new_quote = new_quote + newer_x + " "
    return new_quote.strip()


def quote_pruner(context, quotes, movies):
    """We don't want useless or meaningless quotes so we prune them out.

    Arguments
    =========

    context: a list of strings

    quotes : a list of strings

    movies : a list of strings

    Returns
    =======

    context, quotes, movies : a pruned context, movies, and quotes, based on quote pruning
    """
    kill_sign = "asdf no good this quote is toodleloos"
    single_list = (".", "/", ":", ",", ";", "!", "?", "(", ")", '"', "-", "]", "'", "[", "#", "$",
                   "well", "what", "why", "goodnight", "hello", "how", "no", "yes", "yep", "nope", "pick", "cigarette",
                   "obviously", "kay", "scrappy", "good", "bad", "great", "goodbye", "who", "hoke", "yes'm", "come",
                   "ouch", "huh", "shit", "ed", "fuck", "oh", "right", "ebay", "nothing", "me", "you", "mount", "pray",
                   "sometimes",
                   "really", "ditto", "jeez", "exactly", "bull", "bullshit", "yep", "bing", "38", "occupation",
                   "pyscho", "ok",
                   "okay", "ooh", "dance", "terrific", "cuban", "mexican", "but", "blue", "wood", "apples", "cider",
                   "exactly",
                   "david", "fire", "y'all", "so", "always", "hey")

    index_list = []

    for x in range(len(quotes)):
        y = quotes[x].split()
        if len(y) == 1:
            no_punct = punct_strip(y[0])
            if (no_punct in single_list) or (len(no_punct) == 0):
                context[x] = kill_sign
                quotes[x] = kill_sign
                movies[x] = kill_sign

    # Remove all kill signs
    while (kill_sign in movies):
        context.remove(kill_sign)
        quotes.remove(kill_sign)
        movies.remove(kill_sign)

    return context, quotes, movies


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
            norms[i] += math.pow((tf * idf[term]), 2)

    norm_array = numpy.array(norms)
    return numpy.sqrt(norm_array)


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

    max_dself.ratio: float,
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
        if df >= min_df and float(df / n_docs) <= max_df_ratio:
            idf[term] = math.log(float(n_docs / (1 + df)))
    return idf


if __name__ == "__main__":
    context_file = "final_context_1.json"
    movie_file = "final_movies_1.json"
    quote_file = "final_quotes_1.json"
    year_rating_file = "final_year_rating_1.json"

    context = read_file(context_file)
    movies = read_file(movie_file)
    quotes = read_file(quote_file)
    year_rating_dict = read_file(year_rating_file)

    # Reincode to unicode
    for i in range(len(context)):
        context[i] = context[i].encode("utf-8").decode("utf-8")
        movies[i] = movies[i].encode("utf-8").decode("utf-8")
        quotes[i] = quotes[i].encode("utf-8").decode("utf-8")

    context, quotes, movies = quote_pruner(context, quotes, movies)

    # Initialize query tokenizer
    tokenizer = TreebankWordTokenizer()
    inverted_index = {}  # Including Stop Words
    for context_id, context in enumerate(context):
        context_tokens = tokenizer.tokenize(context)
        for term in context_tokens:
            if term in inverted_index:
                lst = inverted_index[term]
                found = False
                for i, tup in enumerate(lst):
                    if context_id == tup[0]:
                        lst[i] = (context_id, tup[1] + 1)
                        found = True
                if not found:
                    inverted_index[term].append((context_id, 1))
            else:
                inverted_index[term] = [(context_id, 1)]

    # Compute idf values for each term
    idf = compute_idf(inverted_index, len(context))
    # Prune out values removed by idf
    inverted_index = {key: val for key, val in inverted_index.items() if key in idf}
    # Compute document norms
    norms = compute_doc_norms(inverted_index, idf, len(context))

    with io.open("inverted_index.json", 'w', encoding="utf-8") as fout:
        fout.write(unicode(json.dumps(inverted_index, ensure_ascii=False)))
    with io.open("idf.json", 'w', encoding="utf-8") as fout:
        fout.write(unicode(json.dumps(idf, ensure_ascii=False)))