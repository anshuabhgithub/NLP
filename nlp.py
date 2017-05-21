import nltk
from nltk import word_tokenize
from nltk.stem.snowball import SnowballStemmer
import networkx
import matplotlib.pyplot as plt

data = open('script.txt').read()

type(data)

token  = word_tokenize(data)

tag = nltk.pos_tag(token)

pos_of_interest = ['JJ','JJR','JJS','NN','NNP','NNS','NNPS']

def find_word_of_interest(tag ,pos_of_inter):
    stemmer = SnowballStemmer("english")
    data = [item[0] for item in tag if item[1] in pos_of_inter]
    # data1= [stemmer.stem(item.lower()) for item in data if item not in nltk.corpus.stopwords.words('english')]
    data1= [item.lower() for item in data if item not in nltk.corpus.stopwords.words('english')]
    # data1 = data
    data =  set(data1)
    print(len(data1))
    print(len(data))
    return data1

fltr_word = find_word_of_interest(tag,pos_of_interest)
# def freq_rank(data):


def rank_word(data):
    print(len(data))
    graph = networkx.Graph()
    graph.add_nodes_from(set(data))
    length  = len(data)
    window_size = 5
    for i in range(length):
        for j in range(window_size):
            if j+i+1>length:
                break
            else:
                graph.add_edge(data[i],data[j+1])
    # networkx.draw(graph)
    # plt.show()
    ranks = networkx.pagerank(graph,max_iter=10000,alpha=0.9)
    rank = sorted(ranks.items(), key=lambda x: x[1], reverse=True)
    # rank  = ranks
    ranks  = dict(rank)
    return ranks

ranks = rank_word(fltr_word)
key=list(ranks.keys())
print(key[:10])
rank = list(ranks)
# print(rank[:10])
