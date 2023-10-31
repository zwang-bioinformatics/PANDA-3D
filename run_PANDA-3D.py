import sys;
tag_ = 'panda3dcode/'
sys.path.append(tag_)

import numpy as np
import pandas as pd
import sys,os
import datetime
import pickle
import copy
from copy import deepcopy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn import Sequential as Seq, Linear as Lin, ReLU
import torch.nn.functional as F
from math import sqrt
import json
import time
from argparse import Namespace


from gvp_transformer import GVPTransformerModel # modified Model
from data import Alphabet, Alphabet_goclean  # modified token for GO terms
import esm
from model_util import get_n_params, CreateDataset_Server, BatchGvpesmConverter
from torch.utils.data import DataLoader

def predicate(model, device, dataloader, prediction_go_mask):
    model.eval()
    loss_batch = []
    protein, prediction = [], []
    with torch.no_grad():
        for i,data in enumerate(dataloader):
            model_out = model(data['esm1vs'],data['coords'], data['seqs'], data['padding_mask'], data['plddts'],
                        return_all_hiddens=False)
            logits = torch.squeeze(model_out[0],1)
            protein.append(data['proteins'][0])
            prediction_ = torch.sigmoid(logits).cpu().detach().numpy()[:,prediction_go_mask]
            prediction.append(prediction_[0])
    test_df = pd.DataFrame({'protein': protein,
                            'prediction': prediction})
    return test_df

def prediction2text(alphabet, predictions, prediction_format_f):
    prediction_format_fh = open(prediction_format_f, 'w')
    prediction_format_fh.write(f'AUTHOR PANDA-3D\nMODEL 1\nKEYWORDS sequence embedding, geometric vector perceptron, transformer.\n')
    for i, row in enumerate(predictions.itertuples()):
        target = row.protein
        scores = row.prediction
        # sort scores
        scores_dic = {}
        for i,score in enumerate(scores):
            score = round(score,2)
            if score > 1 or score<= 0.09:
                continue
            token = alphabet.all_toks[i]
            if 'GO' in token:
                scores_dic[token] = score
        scores_sorted = dict(sorted(scores_dic.items(), key=lambda item: item[1],reverse=True))
        # save scores
        for key in scores_sorted:
            prediction_format_fh.write(f'{target}\t{key}\t{"%.2f" % scores_sorted[key]}\n')    
    prediction_format_fh.write('END\n')
    prediction_format_fh.close()

if True:
    # python train.py
    start = time.time()
    args_f = f'{tag_}/args.json'
    model_args = json.load(open(args_f,'r'))
    model_args = Namespace(**model_args)
    # use cpu for testing
    model_args.device = ['cuda:0']

    alphabet = Alphabet.from_architecture(model_args.arch)
    terms = list(pd.read_pickle(tag_ + model_args.terms_pkl))
    alphabet_go = Alphabet_goclean(terms)

    input_dir = sys.argv[1]
    out_model = tag_ + f'trained.model'
    out_dir = f'{input_dir}'
    out_file = f'{out_dir}/prediction.txt'
    
    # all tokens: GO-> True; others-> False
    prediction_go_mask = np.array(['GO' in tok for tok in alphabet_go.all_toks])
    
    model = GVPTransformerModel(
                model_args,
                alphabet,
                alphabet_go,
        )
    model.load_state_dict(torch.load(out_model,map_location='cuda:0'))
    model.to(model_args.device[0])
    
    test_data = CreateDataset_Server(input_dir, batch_size=1)
    batch_gvpesm_converter = BatchGvpesmConverter(alphabet, alphabet_go, model_args.coords_mask_plddt_th, model_args.device[0])
    test_dataloader = DataLoader(test_data, batch_size= 1, shuffle=True, collate_fn=batch_gvpesm_converter)
    test_df = predicate(model, model_args.device[0], test_dataloader, prediction_go_mask)
    prediction2text(alphabet_go, test_df,  out_file)
    #print(f'done, save to {out_file}')
