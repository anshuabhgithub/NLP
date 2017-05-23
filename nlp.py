import nltk
from nltk import word_tokenize
from nltk.stem.snowball import SnowballStemmer
import networkx
import matplotlib.pyplot as plt
from nltk.corpus import wordnet as wn
import pickle

data = open('script.txt').read()

type(data)

token  = word_tokenize(data)

tag = nltk.pos_tag(token)

pos_of_interest = ['jj','JJR','JJS','NN','NNP','NNS','NNPS']

verb = ['VBD','VBP','VBN','VB','VBZ','VVG']
adjective = ['RBR','JJR','JJ','JJS']
noun= ['NNPS','NN','NNS']
pronoun = ['PRP']


### function to find the word of interest in Document
def find_word_of_interest(tag ,pos_of_inter):

    stemmer = SnowballStemmer("english")

    data = [item[0] for item in tag if item[1] in pos_of_inter]

    data1= [item.lower() for item in data if item.isalpha()]

    data = [item for item in data1 if item not in nltk.corpus.stopwords.words('english')]

    return data

### function to filter the tagged data
def fltr_tag(data,pos_of_interest):

    fltr_tag = [item for item in tag if item[1] in pos_of_interest]

    flt_tag = [item for item in fltr_tag if item[0] not in nltk.corpus.stopwords.words('english')]

    stemmed_tag=[]

    for item in flt_tag:
        if item[1]  in verb:
            temp = wn.morphy(item[0].lower(), wn.VERB)
            if temp ==None:
                continue
            stemmed_tag.append((temp,item[1]))

        if item[1]  in noun:

            temp= wn.morphy(item[0].lower(), wn.NOUN)

            if temp ==None:
                temp = item[0]
            stemmed_tag.append((temp,item[1]))
            # if item[0] == 'foods':
                # print("before stemming " + item[0] + " after stemming "+ temp)
        tag.append(item)
    return stemmed_tag


## function to rank the word given the sequence of word in Document
def rank_word(data):
    # print(len(data))
    graph = networkx.Graph()
    graph.add_nodes_from(set(data))
    length  = len(data)
    window_size = 3
    count =0;

    for i in range(length):
        for j in range(window_size):
            if (j+i+1)>=length:
                break
            else:
                count+=1;
                graph.add_edge(data[i],data[i+j+1])
    networkx.draw_networkx(graph, pos=None, arrows=True, with_labels=True)

    ranks = networkx.pagerank(graph,max_iter=10000,alpha=0.9)

    rank = sorted(ranks.items(), key=lambda x: x[1], reverse=True)

    ranks  = dict(rank)

    return ranks



def get_rank():
    tagg = fltr_tag(tag,noun+adjective+verb)
    fltr_word = find_word_of_interest(tagg,pos_of_interest)
    ranks = rank_word(fltr_word)
    key=list(ranks.keys())
    print(key[:15])
    return key

## function to calculate the semantic similarity
def word_sim(word1,word2,pos1,pos2):
    if pos1=='VERB':
        w1 = wn.synsets(word1,pos = wn.VERB)
    if pos1=='NOUN':
        w1 = wn.synsets(word1,pos =wn.NOUN)
    if pos2=='VERB':
        w2 = wn.synsets(word2,pos = wn.VERB)
    if pos2=='NOUN':
        w2 = wn.synsets(word2,pos =wn.NOUN)

    if (len(w1) != 0)&(len(w2)!=0):
        d1 =wn.synset(w1[0].name())
        d2 = wn.synset(w2[0].name())
        return d1.wup_similarity(d2)
