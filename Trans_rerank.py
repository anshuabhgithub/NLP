import nltk
from nltk import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import wordnet as wn
import networkx
import matplotlib.pyplot as plt
import pickle
import nlp
import time

start = time.time()


tag=[]

for i in range(1,4):
    data = open('transcript_'+str(i)+'.txt').read()
    token  = word_tokenize(data)
    tag = tag+(nltk.pos_tag(token))




## part of speach of interest


verb = ['VBD','VBP','VBN','VB','VBZ','VVG']
adjective = ['RBR','JJR','JJ','JJS']
noun= ['NNPS','NN','NNS']
pronoun = ['PRP']

pos_of_interest = noun + adjective + verb


fltr_tag = nlp.fltr_tag(tag,pos_of_interest)


###function to calculate the semantic similarity between two words

def word_similarity(word1,word2,tag):
    sim=[0]
    if tag in verb:
        if len(wn.synsets(word1,pos = wn.VERB)) > 0:
            temp =nlp.word_sim(word1,word2,'VERB','VERB')
            if temp != None:
                sim.append(temp)
        if len(wn.synsets(word1,pos = wn.NOUN)) > 0:
            temp= nlp.word_sim(word2,word1,'VERB','NOUN')
            if temp != None:
                sim.append(temp)
    if tag in noun:
        if len(wn.synsets(word1,pos = wn.VERB)) > 0:
            temp = nlp.word_sim(word1,word1,'VERB','NOUN')
            if temp != None:
                sim.append(temp)
        if len(wn.synsets(word1,pos = wn.NOUN)) > 0:
            temp =nlp.word_sim(word1,word2,'NOUN','NOUN')
            if temp != None:
                sim.append(temp)
    return max(sim);




### function to compare top n word with transcript word
def compare_word(N):
    temp_tag = dict(fltr_tag)
    rank_file = open('rank.pkl','rb')
    rank = pickle.load(rank_file)
    rank_file.close()
    out_file = open("word_similarity_matrix.txt",'w')
    if len(rank) ==0:
        print("rank the word first")
        return
    else:
        out_file = open("similarity_out.txt",'w')
        for rk in rank[:N]:
            out_file.write('{}'.format("word semantic similarity score to ** "+rk+" **is \n"))
            score = {}
            print(rk)
            for item in temp_tag.items():
                temp= word_similarity(rk,item[0],item[1])
                if temp>.5:
                    score[item[0]] = temp
            out_file.write('{}'.format(score))
            out_file.write('{}'.format("\n"))
            print(score)




## function to rerank top N word based on Document given
def re_rank(N,rank):
    fltr_word = nlp.find_word_of_interest(fltr_tag,noun+adjective)
    ranks = nlp.rank_word(fltr_word)
    new_rank = {}
    for item in rank[:N]:
        new_rank[item] = ranks[item]
    new_rank = sorted(new_rank.items(), key=lambda x: x[1], reverse=True)
    # key=list(ranks.keys())
    print(list(dict(new_rank).keys()))
    return new_rank
