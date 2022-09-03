from input_data import InputData
import argparse
from model import SkipGramModel
from torch.autograd import Variable
import torch
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
import random
import numpy as np
import os

class Word2Vec:
    def __init__(self, input_file_name, output_file_name , output_model_dir, output_dir, emb_dimension=300, batch_size=50,
                 window_size=5, iteration=5, initial_lr=0.025, neg_num=5, min_count=5):

        self.data = InputData(input_file_name, min_count)
        self.output_file_name = output_file_name
        self.output_model_dir = output_model_dir
        self.output_dir = output_dir

        self.emb_size = len(self.data.word2id)
        self.emb_dimension = emb_dimension
        self.batch_size = batch_size
        self.window_size = window_size
        self.iteration = iteration
        self.initial_lr = initial_lr
        self.neg_num = neg_num
        self.skip_gram_model = SkipGramModel(self.emb_size, self.emb_dimension)
        self.skip_gram_model.cuda()
        self.optimizer = optim.SGD(self.skip_gram_model.parameters(), lr=self.initial_lr)

    def train(self):
        
        START = 0 
        if not os.path.exists(self.output_dir+"/log_final.txt"):
            if os.path.exists(self.output_dir+"/log_temp.txt"):
                with open(self.output_dir+"/log_temp.txt","r") as f:
                    log_dic0 = eval(f.readlines()[0])
                    START = log_dic0['laststep']
                    restore_checkpoint = log_dic0["lastcheckpoint"]
                    self.skip_gram_model.load_state_dict(torch.load(restore_checkpoint))
                    print(f"Restore from the checkpoint:{restore_checkpoint}! Last Step:{START}")
        
        else:
            print("!!!! Please remove the log_final.txt !!!")
            exit(0)

        pair_count = self.data.evaluate_pair_count(self.window_size)
        batch_count = self.iteration * pair_count / self.batch_size
        process_bar = tqdm(range(int(batch_count-START)))

        count = int(batch_count) // 30
        for i0 in process_bar:
            i= i0 + START
            pos_pairs = self.data.get_batch_pairs(self.batch_size, self.window_size)
            neg_v = self.data.get_neg_v_neg_sampling(pos_pairs, self.neg_num)

            pos_u = [pair[0] for pair in pos_pairs]
            pos_v = [pair[1] for pair in pos_pairs]
            u_bert = [pair[2] for pair in pos_pairs]

            pos_u = Variable(torch.LongTensor(pos_u)).cuda()
            pos_v = Variable(torch.LongTensor(pos_v)).cuda()
            neg_v = Variable(torch.LongTensor(neg_v)).cuda()

            self.optimizer.zero_grad()
            loss = self.skip_gram_model.forward(pos_u, pos_v, neg_v, u_bert,)
            loss.backward()
            clip_grad_norm_(self.skip_gram_model.parameters(), max_norm=5)
            self.optimizer.step()

            process_bar.set_description("Loss: %0.8f, lr: %0.6f" %
                                        (loss.item(), self.optimizer.param_groups[0]['lr']))

            if i * self.batch_size % 100000 == 0:
                lr = self.initial_lr * (1.0 - 1.0 * i / batch_count)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr

            if i != 0 and i % count == 0:
                log_dic = dict()
                self.skip_gram_model.save_embedding(self.data.id2word, self.output_file_name + "data.emb300." +str(i) +'_'+ str(self.initial_lr))
                torch.save(self.skip_gram_model.state_dict(), self.output_model_dir+"checkpoint_" + str(i) +".pt")
                log_dic["laststep"] = i
                log_dic["lastcheckpoint"] = self.output_model_dir+"checkpoint_" + str(i) +".pt"
                log_dic["lastembedding"] = self.output_file_name + "data.emb300." +str(i) +'_'+ str(self.initial_lr)
                
                with open(self.output_dir+"/log_temp.txt","w") as f_log:
                    f_log.writelines(str(log_dic))
                    # print(str(log_dic))
        
        # Save the final embedding
        self.skip_gram_model.save_embedding(self.data.id2word, self.output_dir + "data.emb300." +'_final')
        torch.save(self.skip_gram_model.state_dict(), self.output_dir+"checkpoint_final.pt")

        log_dic_final = dict()
        log_dic_final["finalstep"] = i
        log_dic_final["finalcheckpoint"] = self.output_dir+"checkpoint_final.pt"
        log_dic_final["finalembedding"] = self.output_dir + "data.emb300." +'_final'

        with open(self.output_dir+"/log_final.txt","w") as f_log:
            f_log.writelines(log_dic_final)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", default="../data_common/data_new.txt", type=str)
    parser.add_argument("--output_dir_root", default="../output/bert2vec_mean_contexts_2/", type=str)
    parser.add_argument("--emb_dim", default=300, type=int)
    parser.add_argument("--batch_size", default=1000, type=int)
    parser.add_argument("--window_size", default=5, type=int)
    parser.add_argument("--iteration", default=1, type=int)
    parser.add_argument("--initial_lr", default=0.08, type=float)
    parser.add_argument("--neg_num", default=5, type=int)
    parser.add_argument("--min_count", default=5, type=int)
    parser.add_argument('--seed', type=int, default=12345)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    output_dir = args.output_dir_root + "batch_"+str(args.batch_size)+"/"

    output_file = output_dir +"embeddings/"
    output_model_dir = output_dir +"checkpoints/"
    if not os.path.exists(output_file):
        os.makedirs(output_file)
    if not os.path.exists(output_model_dir):
        os.makedirs(output_model_dir)

    w2v = Word2Vec(args.input_file, output_file, output_model_dir, output_dir, args.emb_dim, args.batch_size, args.window_size, args.iteration,
                   args.initial_lr, args.min_count)
    w2v.train()
