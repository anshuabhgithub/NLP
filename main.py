import argparse
import time
start = time.time()
import nltk
from nltk import word_tokenize
from nltk.stem.snowball import SnowballStemmer
import networkx
import matplotlib.pyplot as plt
import pickle
import nlp
import Trans_rerank as re
import optparse
import argparse



parser = argparse.ArgumentParser()
parser.add_argument("-r", "--rank", type=int,help="rank the word in script")
parser.add_argument("-c", "--compare", type=int,help="compare top n word with transcript word")
parser.add_argument("-rr", "--rerank", type=int,help="rerank top n word base on transcript word")
args = parser.parse_args()

if args.rank != None:
    rank = nlp.get_rank()
    output = open('rank.pkl','wb')
    pickle.dump(rank,output)
    output.close()
    print("top N words are following")
    print(rank[:args.rank])

if args.rerank != None:
        N =args.rerank
        rank_file = open('rank.pkl','rb')
        rank = pickle.load(rank_file)
        rank_file.close()
        print(rank[:N])
        re.re_rank(N,rank)

if args.compare != None:
    re.compare_word(args.compare)
