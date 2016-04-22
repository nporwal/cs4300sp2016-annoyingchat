from .models import Docs
import os
import random
import json


def read_file(path):
	#path = Docs.objects.get(id = n).address;
	#above line gives error "no such table: project_template_docs"
	file = open(path)
	transcripts = json.load(file)
	return transcripts

def find_random(q):
	quotes = read_file("jsons/quotes.json")
	movies = read_file("jsons/movies.json")
	result = []
	r = random.randint(0,len(quotes))
	result.append(quotes[r] + " - \"" + movies[r] + "\"")
	return result

def find_similar(q):
	return ["trivial algorithm placeholder response"]

def find_final(q):
	return ["final algorithm placeholder response"]