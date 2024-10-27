import ssl
from ssl import _create_unverified_context

import nltk
from nltk.corpus import wordnet as wn

from nltk.corpus.reader import Synset

try:
    _create_unverified_https_context = _create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Just in the first run, select the following corpora to download:
# wordnet, stopwords, omw-1.4 (Open Multilingual Wordnet)
# After the first run, comment the following line
# nltk.download()


syn = wn.synsets('car')
print("synsets containing the lemma car", syn)

car: Synset = wn.synset('car.n.01')

print("synset car.n.01 lemma names:", car.lemma_names())

print("synset car.n.01 definition:", car.definition())

print("synset car.n.01 hyponyms:", car.hyponyms())

print("synset car.n.01 hypernyms:", car.hypernyms())

print()
print("Similarities:")

e = wn.synset('motor_vehicle.n.01')

print("car and motor_vehicle:", car.path_similarity(e))

h = wn.synset('horse.n.01')

print("car and horse:", car.path_similarity(h))

renoir = wn.synsets('renoir')
print(renoir)

impressionist = wn.synsets('impressionist')
print(impressionist)

print("renoir and impressionist:", renoir[0].path_similarity(impressionist[0]))


def get_similar_synsets(target_synset1, threshold=0.5):
    similar_synsets1 = []
    for synset in wn.all_synsets():
        similarity = target_synset1.path_similarity(synset)
        if similarity and similarity >= threshold:
            similar_synsets1.append((synset, similarity))
    return similar_synsets1


print()
print("Similar synsets to dog.n.01 with threshold 0.5:")
target_synset = wn.synset('dog.n.01')
similar_synsets = get_similar_synsets(target_synset, threshold=0.5)
for synset, similarity in similar_synsets:
    print(f"{synset.name()}: {similarity}")
