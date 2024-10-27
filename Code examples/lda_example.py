import re
import datetime

from gensim import corpora
from gensim import models
from gensim import similarities
from pprint import pprint  # pretty-printer

from nltk import PorterStemmer
from nltk.corpus import stopwords

init_t: datetime = datetime.datetime.now()

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

# another corpus (example in slide)
documents1 = ["eat turkey on turkey day holiday",
              "i like to eat cake on holiday",
              "turkey trot race on thanksgiving holiday",
              "snail race the turtle",
              "time travel space race",
              "movie on thanksgiving",
              "movie at air and space museum is cool movie",
              "aspiring movie star"]

porter = PorterStemmer()

# remove common words and tokenize
stoplist = stopwords.words('english')
texts = [
    [porter.stem(word) for word in document.lower().split() if word not in stoplist]
    for document in documents
]

# create mapping keyword-id
dictionary = corpora.Dictionary(texts)

id2token = dict(dictionary.items())

# create the vector for each doc
model_bow = [dictionary.doc2bow(text) for text in texts]

# create the LDA model from bow vectors
lda = models.LdaModel(model_bow, num_topics=2, id2word=dictionary, random_state=30)
lda_vectors = []
for v in model_bow:
    lda_vectors.append(lda[v])

print()
print("LDA vectors for docs (in terms of topics):")
i = 0
for v in lda_vectors:
    print(v, documents[i])
    i += 1

matrix_lda = similarities.MatrixSimilarity(lda_vectors)
print()
print("Matrix similarities")
print(matrix_lda)


def convert(match):
    return dictionary.id2token[int(match.group(0)[1:-1])]


print("LDA Topics:")
for t in lda.print_topics(num_words=30):
    print(re.sub('"[0-9]+"', convert, str(t)))

end_creation_model_t: datetime = datetime.datetime.now()

print()

# obtain LDA vector for the following doc
# doc = "Human computer interaction"
doc = "trees graph human"
doc_s = [porter.stem(word) for word in doc.lower().split() if word not in stoplist]

vec_bow = dictionary.doc2bow(doc_s)
vec_lda = lda[vec_bow]

# calculate similarities between doc and each doc of texts using lda vectors and cosine
sims = matrix_lda[vec_lda]

# sort similarities in descending order
sims = sorted(enumerate(sims), key=lambda item: -item[1])

print()
print("Given the doc: " + doc)
print("whose LDA vector is: " + str(vec_lda))
print()
print("The Similarities between this doc and the documents of the corpus are:")
for doc_position, doc_score in sims:
    print(doc_score, documents[doc_position])

end_t: datetime = datetime.datetime.now()

# get execution time
elapsed_time_model_creation: datetime = end_creation_model_t - init_t
elapsed_time_comparison: datetime = end_t - end_creation_model_t
print()
print('Execution time model:', elapsed_time_model_creation, 'seconds')
print('Execution time comparison:', elapsed_time_comparison, 'seconds')
