from gensim import corpora
from pprint import pprint  # pretty-printer
import re

from nltk.corpus import stopwords
from nltk import PorterStemmer


documents = [
    "Human machine survey computer interface interface eps time for lab abc computer applications user",
    "A survey of user opinion of computer system user response time computer user interface interface",
    "The EPS user users interfaces interface human interface computer human management system user",
    "System and human interface interface engineering testing of EPS computer user",
    "Relation of users perceived response time to error measurement trees",
    "The generation of random binary unordered paths minors user user computer",
    "The intersection graph of paths in trees paths trees",
    "Graph minors IV Widths of trees and well quasi ordering graph paths",
    "Graph minors A tree paths binary trees graphs",
]

porter = PorterStemmer()

# remove common words and tokenize
stoplist = stopwords.words('english')
texts = [
    [porter.stem(word) for word in document.lower().split() if word not in stoplist]
    for document in documents
]

print("Tokens of each document:")
pprint(texts)

# create mapping keyword-id
dictionary = corpora.Dictionary(texts)

# create the vector for each doc
model_bow = [dictionary.doc2bow(text) for text in texts]

id2token = dict(dictionary.items())

def convert(match):
    return dictionary.id2token[int(match.group(0)[0:-1])]


print()
print("Vectors for documents (the positions with zeros are not shown):")
for doc in model_bow:
    print(re.sub("[0-9]+,", convert, str(doc)))
