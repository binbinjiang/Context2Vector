import argparse
import gzip
import math
import numpy
import re
import sys

import random

from copy import deepcopy

"""
Simple example showing evaluating embedding on similarity datasets
"""
import logging
from six import PY34, iteritems
from tqdm.utils import FormatReplace
from web.datasets.similarity import fetch_MEN, fetch_WS353, fetch_SimLex999, fetch_RW, fetch_RG65, fetch_SimLex999, fetch_MTurk
from web.datasets.categorization import fetch_AP, fetch_battig, fetch_BLESS, fetch_ESSLI_1a, fetch_ESSLI_2b, fetch_ESSLI_2c
from web.datasets.analogy import fetch_google_analogy, fetch_msr_analogy, fetch_semeval_2012_2


from web.embeddings import fetch_GloVe,load_embedding
from web.evaluate import evaluate_similarity, evaluate_on_all, evaluate_on_semeval_2012_2, evaluate_analogy,evaluate_categorization
from web.embedding import Embedding
from six.moves import cPickle as pickle

# Configure logging
logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.DEBUG, datefmt='%I:%M:%S')

from similarity import Dist2prob

isNumber = re.compile(r'\d+.*')
def norm_word(word):
  if isNumber.search(word.lower()):
    return '---num---'
  elif re.sub(r'\W+', '', word) == '':
    return '---punc---'
  else:
    return word.lower()

''' Read all the word vectors and normalize them '''
def read_word_vecs(filename):
  wordVectors = {}
  if filename.endswith('.gz'): fileObject = gzip.open(filename, 'r')
  else: fileObject = open(filename, 'r', encoding="utf-8")
  
  for line in fileObject:
    line = line.strip().lower()
    word = line.split()[0]
    
    try:
      vec = numpy.array(line.split()[1:], dtype=numpy.float64)
    except:
      continue

    wordVectors[word] = numpy.zeros(len(line.split())-1, dtype=numpy.float64)
    # for index, vecVal in enumerate(line.split()[1:]):
    #   # wordVectors[word][index] = float(vecVal)

    wordVectors[word] = vec

    ''' normalize weight vector '''
    wordVectors[word] /= math.sqrt((wordVectors[word]**2).sum() + 1e-6)
    
  sys.stderr.write("Vectors read from: "+filename+" \n")
  return wordVectors

''' Write word vectors to file '''
def print_word_vecs(wordVectors, outFileName):
  sys.stderr.write('\nWriting down the vectors in '+outFileName+'\n')
  with open(outFileName, 'w', encoding="utf-8") as outFile: 
    for word, values in wordVectors.items():
      outFile.write(word+' ')
      val_arr = '' 
      for val in wordVectors[word]:
        val_arr = val_arr + str(val) + ' ' 
        # outFile.write('%.4f' %(val)+' ')
        # outFile.write(f'{val}+' ')
      outFile.writelines(val_arr.strip()+'\n') # .strip() for deleting the last space
  
''' Read the PPDB word relations as a dictionary '''
def read_lexicon(filename):
  lexicon = {}
  for line in open(filename, 'r', encoding="utf-8"):
    words = line.lower().strip().split()
    lexicon[norm_word(words[0])] = [norm_word(word) for word in words[1:]]
  return lexicon


''' Retrofit word vectors to a lexicon '''
def tsne_retrofit(wordVecs, lexicon, numIters, v_input = 100):
  newWordVecs = deepcopy(wordVecs)
  wvVocab = set(newWordVecs.keys())
  loopVocab = wvVocab.intersection(set(lexicon.keys()))

  
  myDist2prob = Dist2prob(v_input)
  

  for it in range(numIters):
    # loop through every node also in ontology (else just use data estimate)

    for word in loopVocab:
      # print("a:",lexicon[word])
      # print("b:",set(lexicon[word]))

      wordNeighbours = set(lexicon[word]).intersection(wvVocab)
      numNeighbours = len(wordNeighbours)
      #no neighbours, pass - use data estimate
      
      if numNeighbours == 0:
        continue
      # the weight of the data estimate if the number of neighbours
      # newVec = numNeighbours * wordVecs[word]

      # print(word, wordNeighbours, numNeighbours, wordVecs[word], newVec)
      # print(numNeighbours)

      wordNeighbours_array = numpy.array([newWordVecs[ppWord] for ppWord in wordNeighbours])
      assert wordNeighbours_array.shape[1]==300
      # print("wordNeighbours_array.shape:",wordNeighbours_array.shape) # (N, 300) N vectors
      input_center_vec_np = numpy.expand_dims(wordVecs[word], axis=0)
      # print("input_center_vec_np.shape:",input_center_vec_np.shape) # (1, 300)

      neighbor_vocs =  wordNeighbours_array
      input_center_vec = input_center_vec_np

      distance_weights = myDist2prob.forward(input_center_vec=input_center_vec, neighbor_vocs=neighbor_vocs)
      distance_weights_T = distance_weights.T
      # print(distance_weights.shape) # (1,N)
      # print(distance_weights_T.shape) # (1,N)

      # print(distance_L2_temp_avg/distance_L2_sum/2)
      distance_avg_temp = distance_weights_T*neighbor_vocs 
      distance_avg = numpy.sum(distance_avg_temp,axis=0)
      assert wordVecs[word].shape==distance_avg.shape
      newWordVecs[word] = (distance_avg + wordVecs[word])/2
      # print((distance_weights_T*neighbor_vocs).shape)
      # newWordVecs[word] = neighbor_vocs/2 +wordVecs[word]/2

      # print("newWordVecs[word].shape:",newWordVecs[word].shape)

  return newWordVecs


if __name__=='__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("-i", "--input", type=str, default=None, help="Input word vecs")
  parser.add_argument("-l", "--lexicon", type=str, default=None, help="Lexicon file name")
  parser.add_argument("-o", "--output", type=str, help="Output word vecs")
  parser.add_argument("-n", "--numiter", type=int, default=10, help="Num iterations")
  args = parser.parse_args()

  wordVecs = read_word_vecs(args.input)
  numIter = int(args.numiter)
  
  lexicon = read_lexicon(args.lexicon)


  # v_input is set to 100 by default
  # for v_input in [1, 10 ,50, 100, 200]:
  for v_input in [100]:
    print(f"v_input:{v_input}")

    outFileName = args.output + "_v_input_"+str(v_input)+".txt"

    ''' Enrich the word vectors using ppdb and print the enriched vectors '''
    print_word_vecs(tsne_retrofit(wordVecs, lexicon, numIter, v_input), outFileName)

    fname = outFileName
    results_display = {}

    w_custome = load_embedding(fname,
                    format="glove",
                    normalize=True, lower=False, clean_words=False,
                    load_kwargs={"vocab_size": 159994, "dim": 300})

    tasks = {
            "WS353": fetch_WS353(),
            "WS353S": fetch_WS353(which="similarity"),
            "WS353R": fetch_WS353(which="relatedness"),
            "SimLex999": fetch_SimLex999(),
            "RW": fetch_RW(),
            "MEN": fetch_MEN(),
            "RG65": fetch_RG65(),
            # "MTurk": fetch_MTurk(),
        }

    # Calculate results using helper function
    for name, data in iteritems(tasks):
        results_display[name] = evaluate_similarity(w_custome, data.X, data.y)
        print("Spearman correlation of scores on {} {}".format(name, results_display[name]))

    result_txt = args.output + "v_input_"+str(v_input)+".result.txt"
    with open(result_txt,"w") as f:
      f.writelines(str(results_display))
