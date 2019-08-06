import sys
import time
import random
import datetime
import quepy

from SPARQLWrapper import SPARQLWrapper, JSON

sparql = SPARQLWrapper("http://dbpedia.org/sparql")

dbpedia = quepy.install("dbpedia")