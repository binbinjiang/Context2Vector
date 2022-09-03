from functools import total_ordering
import numpy
from collections import deque
from flair.embeddings import BertEmbeddings
from flair.data import Sentence
import os
import random
import torch.nn as nn
import torch


class InputData:
    """Store data for word2vec, such as word map, sampling table and so on.

    Attributes:
        file_name: path of input file.
        min_count: minimum word frequency required, i.e., word frequency should greater than min_count.
        window_size: size of window.
    """
    
    def __init__(self, file_name, min_count):
        self.input_file_name = file_name
        self.get_words(min_count)
        self.word_pair_catch = deque()
        self.init_sample_table()
        self.embedding = BertEmbeddings(bert_model_or_path = 'bert-base-uncased', layers = '-1', pooling_operation='mean')

    def get_words(self, min_count):
        # paths of files 
        
        root_path = "../data_common/CorpusStatistics"
        # root_path = "../data_common/CorpusStatistics_test"
        basic_file = "basic_statistics.txt" # words_total; vocab_length; sentence_count; words_total_original; min_count
        word_frequency_file = "word_frequency.txt" # word_frequency;
        word_frequency_original_file = "word_frequency_original.txt" # word_frequency_original;
        word2id_file = "word2id.txt" # word2id;
        id2word_file = "id2word.txt" # id2word;

        Load_Boolean = True # decide whether to read original data

        if not os.path.exists(root_path):
            Load_Boolean = False
            os.mkdir(root_path)
        else:
            file_arr = [basic_file, word_frequency_file, word_frequency_original_file, word2id_file, id2word_file]
            actual_file_arr = os.listdir(root_path)
            for f in file_arr:
                if f not in actual_file_arr:
                    Load_Boolean = False

        if not Load_Boolean:
            # self.input_file = open(self.input_file_name)
            self.input_file = []
            with open(self.input_file_name, 'r') as infile:

                self.words_total_original = 0    # the number of words
                self.sentence_count = 0 # the number of all sentences in the file
                self.word_frequency_original = dict()  # original word frequency (uncleaned)

                for line in infile:
                    if line is None or line == '':
                        continue
                    else:
                        self.input_file.append(line)

                    self.sentence_count += 1
                    line = line.strip().split(' ')
                    self.words_total_original += len(line)
                    for w in line:
                        try:
                            self.word_frequency_original[w] += 1
                        except:
                            self.word_frequency_original[w] = 1
                random.shuffle(self.input_file)
                self.input_file_iter = iter(self.input_file)
            
            print('The Total Number of Words (Un-cleaned): %d' % (self.words_total_original))
            print('Vocabulary Length (Un-Cleaned): %d' % len(self.word_frequency_original))

            self.word2id = dict()  # word => id
            self.id2word = dict()  # id => word
            wid = 0 # word id
            self.words_total = self.words_total_original
            self.word_frequency = dict()  # cleaned word frequency, id => word_frequency
            for w, c in self.word_frequency_original.items(): # w:word; c: count
                if c < min_count:
                    self.words_total -= c
                    continue
                self.word2id[w] = wid
                self.id2word[wid] = w
                self.word_frequency[wid] = c
                wid += 1
            # self.word_count = len(self.word2id)
            self.vocab_length = len(self.word2id) # Vocabulary length, i.e., the number of all different words
            assert len(self.word2id)==len(self.id2word)
            assert len(self.word2id)==len(self.word_frequency)

            print('The Total Number of Words (Cleaned): %d' % (self.words_total))
            print('Vocabulary Length (Cleaned): %d' % self.vocab_length)
            print('The Total Number of sentence: %d' % (self.sentence_count))

            # write files
            with open(os.path.join(root_path, basic_file),"w") as f:
                temp_a = dict()
                temp_a["words_total"] = self.words_total
                temp_a["vocab_length"] = self.vocab_length
                temp_a["sentence_count"] = self.sentence_count
                temp_a["words_total_original"] = self.words_total_original
                temp_a["min_count"] = min_count
                f.writelines(str(temp_a))
            with open(os.path.join(root_path, word_frequency_original_file),"w") as f:
                f.writelines(str(self.word_frequency_original))
            with open(os.path.join(root_path, word_frequency_file),"w") as f:
                f.writelines(str(self.word_frequency))
            with open(os.path.join(root_path, word2id_file),"w") as f:
                f.writelines(str(self.word2id))
            with open(os.path.join(root_path, id2word_file),"w") as f:
                f.writelines(str(self.id2word))

        else:
            print("Loading statistics from the local...")
            with open(os.path.join(root_path, basic_file),"r") as f:
                line = f.readline()
                temp_a = eval(line) # str => dict
                self.words_total = temp_a["words_total"]
                self.vocab_length = temp_a["vocab_length"]
                self.sentence_count = temp_a["sentence_count"]
                self.words_total_original = temp_a["words_total_original"]
                self.min_count = temp_a["min_count"]
                
            with open(os.path.join(root_path, word_frequency_original_file),"r") as f:
                self.word_frequency_original = eval(f.readline())

            with open(os.path.join(root_path, word_frequency_file),"r") as f:
                self.word_frequency = eval(f.readline())

            with open(os.path.join(root_path, word2id_file),"r") as f:
                self.word2id = eval(f.readline())

            with open(os.path.join(root_path, id2word_file),"r") as f:
                self.id2word = eval(f.readline())

            print('The Total Number of Words (Un-cleaned): %d' % (self.words_total_original))
            print('Vocabulary Length (Un-Cleaned): %d' % len(self.word_frequency_original))
            print('The Total Number of Words (Cleaned): %d' % (self.words_total))
            print('Vocabulary Length (Cleaned): %d' % self.vocab_length)
            print('The Total Number of sentence: %d' % (self.sentence_count))

            print("test word2id, word=is, id=",self.word2id["is"])
            print("test id2word, id=100, word=",self.id2word[100])
            print("test word_frequency, id=100, word frequency=",self.word_frequency[100])

            #for the usage of get_batch_pairs()
            # self.input_file = open(self.input_file_name)
            self.input_file = []
            with open(self.input_file_name, 'r') as infile:
                for line in infile:
                    if line is None or line == '':
                        continue
                    else:
                        self.input_file.append(line)
            random.shuffle(self.input_file)
            self.input_file_iter = iter(self.input_file)


    def init_sample_table(self):
        self.sample_table = []
        sample_table_size = 1e8
        pow_frequency = numpy.array(list(self.word_frequency.values()))**0.75
        words_pow = sum(pow_frequency)
        ratio = pow_frequency / words_pow
        count = numpy.round(ratio * sample_table_size)
        for wid, c in enumerate(count):
            self.sample_table += [wid] * int(c)
        self.sample_table = numpy.array(self.sample_table)

        print("sample_table is initialized!")


    # @profile
    def get_batch_pairs(self, batch_size, window_size):
        while len(self.word_pair_catch) < batch_size:
            # sentence = self.input_file.readline()

            try:
                sentence = next(self.input_file_iter)
            except:
                del self.input_file_iter
                self.input_file_iter = iter(self.input_file)
                sentence = next(self.input_file_iter)

            # if sentence is None or sentence == '':
            #     self.input_file = open(self.input_file_name)
            #     sentence = self.input_file.readline()

            word_ids = []
            for word in sentence.strip().split(' '):
                try:
                    word_ids.append(self.word2id[word])
                except:
                    continue

            sents = [self.id2word[ids] for ids in word_ids]
            sent = ' '.join(sents)
            # print(sentence)
            try:
                sentence_new = Sentence(sent)
                self.embedding.embed(sentence_new)
            except:
                continue
            
            sentence_new_emb_arr = torch.zeros(len(word_ids), 1, 768)
            for i, _ in enumerate(word_ids):
                u_bert_tmep = sentence_new[i].embedding.unsqueeze(0)
                sentence_new_emb_arr[i] = u_bert_tmep
                assert u_bert_tmep.shape==(1,768)
            # print(len(sentence_new_emb_arr))

            for i, u in enumerate(word_ids):
                # u_bert = sentence_new[i].embedding.unsqueeze(0)
                # u_bert = sentence_new_emb_arr[i]

                # contexts_u_bert_all = sentence_new_emb_arr[max(i - window_size, 0):min(i + window_size, len(word_ids))]
                # contexts_u_bert_all[i] = 0

                # contexts_u_bert_mean = torch.mean(contexts_u_bert_all, dim=0)
                # assert contexts_u_bert_mean.shape==u_bert.shape
                
                contexts_u_bert_all_no_i = torch.cat((sentence_new_emb_arr[max(i - window_size, 0):i],
                                        sentence_new_emb_arr[i+1:min(i + window_size, len(word_ids))]),dim=0)
                u_bert = torch.mean(contexts_u_bert_all_no_i, dim=0)/2 + sentence_new_emb_arr[i]/2

                for j in range(max(i - window_size, 0), min(i + window_size, len(word_ids))):
                    v = word_ids[j]
                    assert u < self.vocab_length
                    assert v < self.vocab_length
                    if i == j:
                        continue
                    
                    self.word_pair_catch.append((u, v, u_bert))

        batch_pairs = []
        for _ in range(batch_size):
            batch_pairs.append(self.word_pair_catch.popleft())
        return batch_pairs

    # @profile
    def get_neg_v_neg_sampling(self, pos_word_pair, count):
        neg_v = numpy.random.choice(
            self.sample_table, size=(len(pos_word_pair), count)).tolist()
        return neg_v

    def evaluate_pair_count(self, window_size):
        self.pair_count = self.words_total * (2 * window_size - 1) - (
            self.sentence_count - 1) * (1 + window_size) * window_size
        print("pair count:", self.pair_count)

        return self.pair_count
