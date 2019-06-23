import networkx as net
import csv
import matplotlib.pyplot as plt
import numpy as np

def create_graph(filename):
    # Should be digraph, but won't change it because it might make code not working
    G = net.Graph()

    file = open(filename, mode='r')
    reader = csv.reader(file, delimiter='\t')
    for row in reader:
        #according to readme the format of the row is userid, movieid, rating, timestamp
        user = row[0]
        item = row[1]
        rating = row[2]

        G.add_node((user, 'user'))
        G.add_node((item, 'item'))
        #unique rating node -> giving the same rating to different movies
        #create multiple "rating"nodes
        G.add_node((user, item))
        # edge between user and rating
        G.add_edge((user, 'user'), (user, item))
        # edge between item and rating
        G.add_edge((item, 'item'), (user, item))

    for node in G.nodes:
        neighbors = G[node]
        outlinks = len(neighbors.items())
        if outlinks > 0:
            for n in neighbors:
                w = 1/outlinks
                G.add_edge(node, n, weight=w)

    return G

# vector representing the preferences of nodes for user (1/sum)
def create_personalized_vector(user_id, graph: net.Graph):
    q_vec : dict = {}
    sum = 0
    for node in graph.nodes:
        try:
            is_int = True
            # node = (userid, itemid) or (userid, 'user') or (itemid, 'item').
            int(node[1])
        except:
            is_int = False

        if node == (str(user_id), 'user') or (node[0] == str(user_id) and is_int):
            q_vec[node] = 1
            sum += 1
        else:
            q_vec[node] = 0

    for item in q_vec.items():
        # normalizing with sum of preference nodes.
        key = item[0]
        value = item[1]
        q_vec[key] = value/sum

    return q_vec

def get_items(filename, user_id):
    item_ids = []
    item_id_ratings = {}

    file = open(filename, mode='r')
    reader = csv.reader(file, delimiter='\t')
    for row in reader:
        if row[0] == str(user_id):
            item_ids.append(row[1])
            item_id_ratings[row[1]] = row[2]

    return item_ids, item_id_ratings

def recommend_k(graph, user_id, train_items, k=10):
    q = create_personalized_vector(user_id, graph)
    pr = net.pagerank(graph, personalization=q)
    # sort nodes based on their values (scores) in descending order.
    sorted_pr = sorted(pr.items(), reverse=True, key=lambda kv: kv[1])

    item_ids = []
    for pr in sorted_pr:
        node = pr[0]
        # We only want to recommend items.
        if node[1] == 'item' and \
                len(item_ids) < k and \
                node[0] not in train_items:
            item_ids.append(node[0])

    return item_ids

# maybe stupid way of calculating hits@k
# because, the problem is not to recommend movies based on ratings
# but rather on what the user has seen before
def calculate_hits(sorted_recommended_items, sorted_test_items):
    n = len(sorted_recommended_items)
    hits = 0
    for item in sorted_recommended_items:
        k = 0
        for test_item in sorted_test_items: # dict of itemid and rating.
            if (test_item[0] == item and k <= n) or \
                (test_item[0] == item and test_item[1] == '5'):
                hits += 1
                break
            k += 1

    return hits/n

def calculate_hits_2(recommended_items, test_items):
    n_test_items = len(test_items)
    n_rec_items = len(recommended_items)

    hits = 0
    for rec_item in recommended_items:
        for test_item in test_items:
            if test_item[0] == rec_item:
                hits += 1
    # todo: check if there even exist test items
    # n_rec_items has size 10 and if we only know fx 7 items the user has watched before
    # We average with these 7 items instead.
    if n_test_items < n_rec_items:
        return hits/n_test_items
    else:
        return hits/n_rec_items

if __name__ == "__main__":
    train_filename = "ml-100k/u1.base"
    test_filename = "ml-100k/u1.test"

    train_G = create_graph(train_filename)

    test_item_ids, test_items = get_items(test_filename, 1)
    train_item_ids, train_items = get_items(train_filename, 1)
    sorted_test_items = sorted(test_items.items(), reverse=True, key=lambda kv: kv[1])

    result = recommend_k(train_G, 1, train_item_ids)

    print("HITS/n")
    print(calculate_hits_2(result, sorted_test_items))

    for r in result:
        print(r)

        # paper does NE-aggregation first
        # then Merge
        # algorithm on slides

        # subgraph (instead of NE-aggregation) function in networkx
        # disjoint_union as merge function

        # create 2 subgraphs with subgraph
        # merge them with disjoint_union
        # make recommendations based on the merged graph
        # calculate_hits
        # consider time it takes for naive approach and merge approach

        # maybe just make 1 subgraph (on timestamp) and tell that paper uses multiple
        # and then merge.



