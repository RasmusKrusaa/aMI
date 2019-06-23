import nltk as nltk
from nltk.corpus import stopwords
import networkx as net
import math
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

file = open("text.txt", mode='r')
sentences = file.read()

stop_words = set(stopwords.words('english'))
sent_tokens = nltk.sent_tokenize(sentences)
stemmed_tokens = []

GS_random_walk = net.Graph()

#creating matrix in order to easier calculate probability distribution
#matrix contains cooccurence between sentences. i.e. matrix[1,2] is the cooccurence between sentence 1 and 2.
number_of_sentences = len(sent_tokens)
sim_matrix = np.zeros((number_of_sentences, number_of_sentences))

# Stemming by removing stopwords. i.e. a, an, ., etc.
for token in sent_tokens:
    stemmed_tokens.append(" ".join([w.lower() for w in token.split(' ') if w.lower() not in stop_words]))

# similarities based on co-occurences
def cooccurence_sim(sent1, sent2):
    words_in_sent1 = nltk.word_tokenize(sent1)
    words_in_sent2 = nltk.word_tokenize(sent2)
    overlap = len(list(set(words_in_sent1).intersection(set(words_in_sent2))))
    return overlap / (math.log10(len(words_in_sent1)) + math.log10(len(words_in_sent2)))

# filling sim_matrix with co-occurrence
for i in range(0, len(stemmed_tokens)):
    for j in range(0, len(stemmed_tokens)):
        if i == j:
           sim_matrix[i,j] = 1
        else:
           sim_matrix[i,j] = cooccurence_sim(stemmed_tokens[i], stemmed_tokens[j])

# Creating graph with value on edges equal probability according to row normalization
# In case of dangling nodes (i.e. 0 similarity), we use alpha in networkx
# to handle teleportation.
for row in range(0, sim_matrix.shape[0]):
    GS_random_walk.add_node(sent_tokens[row])
    summed_sim = sum(sim_matrix[row, ])
    for col in range(0, sim_matrix.shape[1]):
        sim = sim_matrix[row, col]
        probability = sim/summed_sim
        GS_random_walk.add_edge(sent_tokens[row], sent_tokens[col], weight=probability)

# Alpha (dampening factor) determines magnitude of following edges and
# 1-alpha determines magnitude of "jumping/teleporting" between nodes.
random_walk_pr = net.pagerank(GS_random_walk, alpha=0.90)

# Sorting pageranks (dictionary -> key-value pair) based on their value.
random_walk_pr = sorted(random_walk_pr.items(), reverse=True, key=lambda kv: kv[1])

# Selecting approx 30% of document to summarize with
summary_size = int(len(random_walk_pr) * 0.3)

print("--- Simpel summary with cooccurence ---")
for i in range(0, summary_size):
    print(random_walk_pr[i])


#K means clustering summarization
#Vectorizing sentences
vectorizer = TfidfVectorizer(stop_words = 'english')
vectorized_sentences = vectorizer.fit_transform(sent_tokens)
vectorized_document = vectorizer.fit_transform([sentences])

#---K means clustering---
#use sqrt(number of sentences)
number_of_clusters = math.floor(math.sqrt(number_of_sentences))
# n_init number of times to initialize centroids.
model = KMeans(n_clusters=number_of_clusters, init='k-means++', n_init=1)
model.fit(vectorized_sentences)

#Graph used to compute pagerank using KMeans clustering
GS_k_means = net.Graph()

#lambda (relative contribution from source cluster and destination cluster)
lambda_param = 0.85

#similarity matrix
sim_matrix_k_means = np.zeros((number_of_sentences, number_of_sentences))

def cosine_similarity(vec_sent1, vec_sent2):
    top = np.inner(vec_sent1, vec_sent2)
    bottom = norm(vec_sent1) * norm(vec_sent2)
    return top / bottom

def belongs_to_cluster(sentence):
    cluster = model.predict([sentence])[0]
    return model.cluster_centers_[cluster]

for i in range(0, number_of_sentences):
    for j in range (0, number_of_sentences):
        if i == j:
            sim_matrix_k_means[i, j] = 0
        else:
            sent_i = vectorized_sentences[i].toarray()[0]
            sent_j = vectorized_sentences[j].toarray()[0]
            fij = cosine_similarity(sent_i, sent_j)
            clusteri = belongs_to_cluster(sent_i)
            clusterj = belongs_to_cluster(sent_j)
            # document is a single sentence of all sentences concatinated
            # cluster is a single sentence of all sentences in that cluster concatinated
            pii = cosine_similarity(clusteri, vectorized_document.toarray()[0])
            pij = cosine_similarity(clusterj, vectorized_document.toarray()[0])
            wi = cosine_similarity(sent_i, clusteri)
            wj = cosine_similarity(sent_j, clusterj)

            sim_matrix_k_means[i, j] = fij * (lambda_param * pii * wi + (1-lambda_param) * pij * wj)

for row in range(0, sim_matrix_k_means.shape[0]):
    GS_k_means.add_node(sent_tokens[row])
    summed_sim = sum(sim_matrix_k_means[row, ])
    for col in range(0, sim_matrix_k_means.shape[1]):
        if summed_sim == 0:
            GS_k_means.add_edge(sent_tokens[row], sent_tokens[col], weight=0)
        else:
            sim = sim_matrix_k_means[row, col]
            probability = sim/summed_sim
            GS_k_means.add_edge(sent_tokens[row], sent_tokens[col], weight=probability)

pr_k_means = net.pagerank(GS_k_means, alpha=0.9)

# Sorting pageranks (dictionary -> key-value pair) based on the values.
pr_k_means_sorted = sorted(pr_k_means.items(), reverse=True, key=lambda kv: kv[1])

# Selecting approx. 30% to summarize document with.
summary_size = int(len(pr_k_means_sorted) * 0.3)

print("--- Summary based on KMeans ---")
for i in range(0, summary_size):
    print(pr_k_means_sorted[i])

file.close()