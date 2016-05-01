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

# Download NLTK stopwords if you haven't already
# nltk.download()


def read_file(path):
    # path = Docs.objects.get(id = n).address;
    # above line gives error "no such table: project_template_docs"
    file = open(path)
    transcripts = json.load(file)
    return transcripts


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


def make_word_set(context):
    """ Computes the set of all words used in a list of strings.

    Arguments
    =========

    context: a list of strings

    Returns
    =======

    word_set: set of distinct words
    """
    tokenizer = TreebankWordTokenizer()
    sw = stopwords.words('english')
    word_list = []
    for string in context:
        tkns = tokenizer.tokenize(string)
        for tk in tkns:
            if tk not in sw:
                word_list.append(tk)
    word_set = set(word_list)
    return word_set


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


# Prune quotes
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


def get_context_words(inv_ind):
    """
    Returns the set of all non-stop-word in the inverted index
    """
    # Get English stop words
    stop_words = stopwords.words('english')

    # Find set of words that aren't stop words and are in our tfidf
    base_words = []
    word_to_index = {}
    for word in inv_ind:
        if word not in stop_words:
            base_words.append(word)
    for i, word in enumerate(base_words):
        word_to_index[word] = i

    # Return word list and mapping
    return base_words, word_to_index


def update_word_counts(lst, word):
    """Update the list for a word co-occurrance matrix

    Arguments
    =========

    lst: the pair (word,count) list from the word occurrance

    word : the word we're updating

    Returns
    =======

    new_list : an updated word occurrance list
    """
    found = False
    new_list = []
    for w, c in lst:
        if word == w:
            found = True
            count = c + 1
            new_list.append((w, count))
            break
        else:
            new_list.append((w, c))
    if not (found):
        new_list.append((word, 1))
    return new_list


class QuoteFinder:
    def __init__(self):
        context_file = "jsons/final_context_1.json"
        movie_file = "jsons/final_movies_1.json"
        quote_file = "jsons/final_quotes_1.json"
        year_rating_file = "jsons/final_year_rating_1.json"

        self.context = read_file(context_file)
        self.movies = read_file(movie_file)
        self.quotes = read_file(quote_file)
        self.year_rating_dict = read_file(year_rating_file)

        # Reincode to unicode
        for i in range(len(self.context)):
            self.context[i] = self.context[i].encode("utf-8").decode("utf-8")
            self.movies[i] = self.movies[i].encode("utf-8").decode("utf-8")
            self.quotes[i] = self.quotes[i].encode("utf-8").decode("utf-8")

        self.context, self.quotes, self.movies = quote_pruner(self.context, self.quotes, self.movies)

        # Initialize query tokenizer
        self.tokenizer = TreebankWordTokenizer()
        self.inverted_index = {}  # Including Stop Words
        for context_id, context in enumerate(self.context):
            context_tokens = self.tokenizer.tokenize(context)
            for term in context_tokens:
                if term in self.inverted_index:
                    lst = self.inverted_index[term]
                    found = False
                    for i, tup in enumerate(lst):
                        if context_id == tup[0]:
                            lst[i] = (context_id, tup[1] + 1)
                            found = True
                    if not found:
                        self.inverted_index[term].append((context_id, 1))
                else:
                    self.inverted_index[term] = [(context_id, 1)]

        # Compute idf values for each term
        self.idf = compute_idf(self.inverted_index, len(self.context))
        # Prune out values removed by idf
        self.inverted_index = {key: val for key, val in self.inverted_index.items() if key in self.idf}
        # Compute document norms
        self.norms = compute_doc_norms(self.inverted_index, self.idf, len(self.context))

        # Check if there is a word co-occurrance file, word count file, and pmi file. if not, remake
        word_co_filename = "jsons/word_co.json"
        word_count_filename = "jsons/word_count_dict.json"
        pmi_dict_filename = "jsons/pmi_dict.json"
        # if ((os.path.isfile(word_co_filename)) and (os.path.isfile(word_count_filename))) and (
        # os.path.isfile(pmi_dict_filename)):
        # Read files
        word_co = read_file(word_co_filename)
        word_count_dict = read_file(word_count_filename)
        pmi_dict = read_file(pmi_dict_filename)
        # else:
        #     # Initialize word co-occurance matrix (merge this with above cell)
        #     # This may take a while
        #     word_list, word_to_index = get_context_words(self.inverted_index)
        #     word_co, word_count_dict = self.find_basic_cooccurance(word_list)
        #     # Get PMI
        #     pmi_dict = self.find_pmi(word_co, word_count_dict)
        #     # Write data
        #     with io.open("word_co.json", 'w', encoding="utf-8") as fout:
        #         fout.write(unicode(json.dumps(word_co, ensure_ascii=False)))
        #     with io.open("word_count_dict.json", 'w', encoding="utf-8") as fout:
        #         fout.write(unicode(json.dumps(word_count_dict, ensure_ascii=False)))
        #     with io.open("pmi_dict.json", 'w', encoding="utf-8") as fout:
        #         fout.write(unicode(json.dumps(pmi_dict, ensure_ascii=False)))

    def find_basic_cooccurance(self, word_list):
        """ Initialize the base word co-occurrance list from our context and quotes.

        Arguments
        =========

        word_list: the list of words which are in our movie space

        Returns
        =======

        word_co : a dictionary representing the word_occurrance matrix
        """
        # Get English stop words
        stop_words = stopwords.words('english')

        # Merge context and quotes
        quote_list = self.quotes
        new_quote_list = []
        for q in quote_list:
            new_q = punct_strip(q)
            if new_q not in self.context:
                new_quote_list.append(new_q)
        context_quotes = self.context + new_quote_list

        # Find co occurances in context data, based co-occurances in a document
        word_co = defaultdict(list)
        word_count_dict = defaultdict(int)
        for doc in context_quotes:
            # Double loop to count word co-occurances
            tkns = self.tokenizer.tokenize(doc)
            for i in range(len(tkns)):
                if tkns[i] not in stop_words:
                    word_count_dict[tkns[i]] += 1
                    for j in range(len(tkns)):
                        if not (j == i) and (tkns[j] in word_list):
                            word_co[tkns[i]] = update_word_counts(word_co[tkns[i]], tkns[j])
        return word_co, word_count_dict

    def update_cooccurance(self, word_co, word_count_dict, word_list, docs):
        """ Updates the word co-occurrance mat and the word count dict with a new set of data.

        Arguments
        =========

        word_co: a word co-occurrance matrix in the form of a dictionary

        word_count_dict: a dictionary that keeps track of the total occurances of a word

        word_list: the list of words which are in our movie space

        docs: the new docs we're using to update our word co-occurance

        Returns
        =======

        word_co, word_count_dict : new word co-occurance dict/mat and new word count dictionary
        """
        # Get English stop words
        stop_words = stopwords.words('english')

        word_co = defaultdict(list)
        # Find co occurances in context data, based on document (content)
        for doc in docs:
            # Double loop to count word co-occurances
            tkns = self.tokenizer.tokenize(punct_strip(doc))
            for i in range(len(tkns)):
                if tkns[i] not in stop_words:
                    for j in range(len(tkns)):
                        if not (j == i) and (tkns[j] in word_list):
                            word_co[tkns[i]] = update_word_counts(word_co[tkns[i]], tkns[j])
        return word_co

    def find_pmi(self, word_co, word_count_dict):
        """ Calculate the pmi of a word based on the word co-occurances

        Arguments
        =========

        word_co: a word occurrance matrix in the form of a dictionary

        word_list: the list of words which are in our movie space

        docs: the new docs we're using to update our word co-occurance

        Returns
        =======
        pmi_dict = a dictionary like word_co but has the pmi's instead
        """
        # PMI(x,y) = log[p(x,y)/(p(x)*p(y))], assuming p(x,y) = co-occurances over total
        pmi_dict = defaultdict(list)

        # Find total
        total = 0
        for key in word_count_dict:
            total += word_count_dict[key]
        total += 1  # smoothing

        # Caclulate PMIs
        for key in word_co:
            p_x = (word_count_dict[key] + 0.0) / total
            pmi_list = []
            lst = word_co[key]
            for word, co in lst:
                p_y = (word_count_dict[word] + 0.0) / total
                p_x_y = (co + 0.0) / (word_count_dict[key] + 1.0 + word_count_dict[word])
                res = math.log(p_x_y / (p_x * p_y + 1.0))
                pmi_list.append((word, res))
            pmi_dict[key] = pmi_list

        return pmi_dict

    def year_rating_weight(self, year, rating, cosine, cur_year=2016, min_year=1925, year_weight=0.3,
                           rating_weight=0.7, cosine_weight=0.8, y_r_weight=0.2):
        """ Compute new score with weighting from the cosine similarity with
        the release year and rating of the movie.

        Arguments
        =========

        year: float, the year of the movie

        rating: float, the rating of the movie (out of 10)

        cosine: float, the cosine similarity of the query against the movie context

        cur_year, min_year: int, current year and minimum year (the lowest year)

        year_weight, rating_weight: the weights for the year and rating

        cosine_weight, y_r_weight: the weights for the cosine sim vs the year and rating value

        Returns
        =======

        a new score that has been weighted with the year and rating of the movie
        """

        w_year = (((cur_year - min_year) - (cur_year - year)) / (cur_year - min_year)) * year_weight
        w_rating = (rating / 10.0) * rating_weight
        return ((w_year + w_rating) * y_r_weight) + (cosine * cosine_weight)

    def query_vectorize(self, q, sw=False):
        # Remove punctuation, lowercase, and encode to utf
        query = punct_strip(q.lower().encode("utf-8").decode("utf-8"))

        # Tokenize query and check query stopword cutoff
        query_words = self.tokenizer.tokenize(query)

        # Remove stop words if necessary
        stop_words = stopwords.words('english')  # Get English stop words
        if (sw):
            new_query = []
            for x in query_words:
                if x not in stop_words:
                    new_query.append(x)
            query_words = new_query

        # Make query tfidf
        query_tfidf = defaultdict(int)
        for word in query_words:
            query_tfidf[word] += 1
        for word in query_tfidf:
            if word in self.idf:
                query_tfidf[word] *= self.idf[word]
            else:
                query_tfidf[word] = 0

        # Find query norm
        query_norm = 0
        for word in query_tfidf:
            query_norm += math.pow(query_tfidf[word], 2)
        query_norm = math.sqrt(query_norm)

        return query_tfidf, query_norm

    def pseudo_rocchio(self, query, relevant, sw=False, a=.3, b=.4, clip=True):
        """
        Arguments:
            query: a string representing the name of the movie being queried for

            relevant: a list of int representing the indices of relevant movies for query

            irrelevant: a list of strings representing the names of irrelevant movies for query

            a,b: floats, corresponding to the weighting of the original query, relevant queriesrespectively.

            clip: boolean, whether or not to clip all returned negative values to 0

        Returns:
            q_mod: a dict representing the modified query vector. this vector should have no negatve
            weights in it!
        """

        relevant_id = []
        for s, i in relevant:
            relevant_id.append(i)

        # vectorize query
        query_tfidf, query_norm = self.query_vectorize(query, sw)

        if query_norm == 0:
            return self.find_random()

        # Calculate alpha*query_vec
        query_vec = query_tfidf
        for word in query_vec:
            query_vec[word] /= query_norm
            query_vec[word] *= a

        # Get words in relevant docs
        relevant_words = []
        relevant_context = []
        for i in relevant_id:
            relevant_context.append(self.context[i])
        for context in relevant_context:
            context_tkns = self.tokenizer.tokenize(context)
            for tkn in context_tkns:
                if tkn not in relevant_words:
                    relevant_words.append(tkn)

        # Collect relevant doc vector sums
        relevant_docs = defaultdict(int)
        for word in relevant_words:
            if word in self.inverted_index:
                for quote_id, tf in self.inverted_index[word]:
                    if quote_id in relevant_id:
                        relevant_docs[word] += (tf / self.norms[quote_id])

        # Calculate beta term
        beta_term = b * (1.0 / len(relevant))
        for key in relevant_docs:
            relevant_docs[key] *= beta_term

        # Sum query and relevant
        q_mod = {k: query_vec.get(k, 0) + relevant_docs.get(k, 0.0) for k in set(query_vec) | set(relevant_docs)}

        # negative checks for terms, if clip
        if (clip):
            for key in q_mod:
                if q_mod[key] < 0:
                    q_mod[key] = 0
            return q_mod
        else:
            return q_mod

    def find_random(self):
        result = []
        r = random.randint(0, len(self.quotes))
        result.append(self.quotes[r] + " - \"" + self.movies[r] + "\"")
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
                    scores[quote_id] += query_tfidf[word] * tf * self.idf[word]

        results = []
        for i, s in enumerate(scores):
            if self.norms[i] != 0:
                results.append((s / (self.norms[i] * query_norm), i))
        results.sort(reverse=True)
        result_quotes = ["{} - \"{}\"".format(self.quotes[i], self.quotes_to_movies[self.quotes[i]]) for _, i in
                         results]
        return result_quotes

    def find_final(self, q, rocchio=True, psuedo_rocchio_num=6, sw=False, pmi_num=10):
        # Vectorize query
        query_tfidf, query_norm = self.query_vectorize(q, sw)

        if query_norm == 0:
            return self.queries

        # Expand query using PMI
        # Use the top x number of co-occurances to expand the query


        # Get scores
        scores = [0 for _ in self.quotes]
        for word in query_tfidf:
            if word in self.inverted_index:
                for quote_id, tf in self.inverted_index[word]:
                    scores[quote_id] += query_tfidf[word] * tf * self.idf[word]

        results = []
        for i, s in enumerate(scores):
            if self.norms[i] != 0:
                results.append((s / (self.norms[i] * query_norm), i))

        # Weight scores with year and rating
        for i in range(len(results)):
            score = results[i][0]
            index = results[i][1]
            year = self.year_rating_dict[self.movies[i]][0]
            rating = self.year_rating_dict[self.movies[i]][1]
            results[i] = (self.year_rating_weight(float(year), float(rating), score), index)

        # sort results
        results.sort(reverse=True)

        if rocchio:
            # Do pseudo-relevance feedback with Rocchio
            mod_query = self.pseudo_rocchio(q, results[:psuedo_rocchio_num], sw)
            mod_query_norm = 0
            for word in mod_query:
                mod_query_norm += math.pow(mod_query[word], 2)
            mod_query_norm = math.sqrt(mod_query_norm)

            # Re-find scores and reweight with year and rating
            scores = [0 for _ in self.quotes]
            for word in mod_query:
                if word in self.inverted_index:
                    for quote_id, tf in self.inverted_index[word]:
                        scores[quote_id] += mod_query[word] * tf * self.idf[word]

            results = []
            for i, s in enumerate(scores):
                if self.norms[i] != 0:
                    results.append((s / (self.norms[i] * mod_query_norm), i))

            # Weight scores with year and rating
            for i in range(len(results)):
                score = results[i][0]
                index = results[i][1]
                year = self.year_rating_dict[self.movies[i]][0]
                rating = self.year_rating_dict[self.movies[i]][1]
                results[i] = (self.year_rating_weight(float(year), float(rating), score), index)

        # Sort and return results
        top_res_num = 5
        results.sort(reverse=True)
        result_quotes = ["{} - {}".format(self.quotes[i], self.movies[i]) for _, i in results[:top_res_num]]
        return result_quotes
