####################
# czhao
####################

import pandas as pd
import numpy as np
import pickle
import glob
import torch
import esm
from torch.utils.data import Dataset
from util import CoordBatchConverter, load_coords
from Bio import SeqIO
def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return 'Number of trainable parameters ' + str(pp//10**6) + 'M'

class CreateDataset(Dataset):
    def __init__(self, database_dir, df_f, batch_size_ = None):
        self.database_dir = database_dir
        self.df_ = pd.read_pickle(database_dir+df_f)
        # Drop not divisible batch size
        #print(len(self.df_),batch_size_,type(batch_size_))
        if (batch_size_ is not None) and (len(self.df_) % batch_size_ != 0):
            self.df_ = self.df_[:-(len(self.df_) % batch_size_)]
        #print(len(self.df_))
    def __len__(self):
        return len(self.df_)

    def load_pick(self, pkl_f):
        with open(pkl_f, 'rb') as handle:
            return pickle.load(handle)
        
    def __getitem__(self, idx):
        protein_id = self.df_.iloc[idx].protein
        feat_path = f'{self.database_dir}/{protein_id[:2]}/{protein_id}.pkl'
        feat_dic = self.load_pick(feat_path)
        feat_dic.pop('label', None)
        return feat_dic


class CreateDataset_Server(Dataset):
    def __init__(self, database_dir, batch_size = None):
        self.database_dir = database_dir
        # load all pdb files
        self.pdb_fs = glob.glob(self.database_dir+'/*pdb')
        # load esm
        self.model, alphabet = esm.pretrained.esm1_t34_670M_UR50S()
        self.batch_converter = alphabet.get_batch_converter()
        # gather features
        proteins, esm1vs, coords, seqs, padding_mask, plddts = [], [], [], [], [], []
        for pdb_f in self.pdb_fs:
            protein_ = pdb_f.split('/')[-1].replace('.pdb','')
            chain_, seq_ = self.pdb_fasta(pdb_f)
            coord_, _ = load_coords(pdb_f, chain_)
            plddt_ = self.parser_plddt(pdb_f)
            esm_ = self.fasta_esm(seq_)
            #print(f'>{protein_}\n{seq_}\n')
            #print(pdb_f, chain_, len(seq_), type(coord_), type(plddt_), type(esm_))
            proteins.append(protein_); esm1vs.append(esm_); coords.append(coord_); seqs.append(seq_); plddts.append(plddt_)
        self.df_ = pd.DataFrame({'protein':proteins, 'esm1v':esm1vs, 'coords':coords, 'seq':seqs, 'plddt':plddts})

    def pdb_fasta(self, f_):
        """assume only one chain in pdb"""
        for record in SeqIO.parse(f_, "pdb-seqres"):
            return record.annotations["chain"], str(record.seq)

    def parser_plddt(self, f_):
        plddts = []
        for line in open(f_,'r').readlines():
            if line[:4] == 'ATOM' and line[13:15] == 'CA':
                plddts.append(np.float32(line[61:66]))
        return np.array(plddts)

    def fasta_esm(self, fasta):
        data = [('',fasta)] #'subunit','fasta'
        batch_labels, batch_strs, batch_tokens = self.batch_converter(data)
        with torch.no_grad():
            results = self.model(batch_tokens, repr_layers=[34])
        token_embeddings = results["representations"][34]
        pred = token_embeddings[0].numpy()[1:]
        return pred

    def __len__(self):
        return len(self.df_)

    def __getitem__(self, idx):
        return self.df_.iloc[idx]


class BatchGvpesmConverter(object):
    """Callable to convert an unprocessed batch to a
    processed (padded feat + labels) batch.
    
    coords: pad inf at ends, nan for diff length
    plddts: pad 0 for ends, -1 for diff length
    esms: pad 0 for ends and diff length
    """
    def __init__(self, alphabet,alphabet_go, coords_mask_plddt_th, device=None):
        self.alphabet = alphabet
        self.alphabet_go = alphabet_go
        self.coords_mask_plddt_th = coords_mask_plddt_th
        self.batch_coord_converter = CoordBatchConverter(self.alphabet)
        self.device = device
            
    def batch_esm_converter(self, esm1vs):
        batch_size = len(esm1vs)
        max_len = max(esm1v.shape[0] for esm1v in esm1vs)
        esm1vs_padded = torch.full((batch_size,max_len+2,esm1vs[0].shape[1]),0.)
        for i, esm1v in enumerate(esm1vs):
            esm1vs_padded[i,1:esm1v.shape[0]+1] = torch.from_numpy(esm1v)
        if self.device is not None:
            esm1vs_padded = esm1vs_padded.to(self.device)
        return esm1vs_padded
    
    def batch_seq_converter(self, seqs):
        seqs_padded_token = []
        max_len = max(len(seq) for seq in seqs)
        for i, seq in enumerate(seqs):
            seq_ = "<go>" + seq + "<eos>" + '<pad>' * (max_len-len(seq))
            seq_token = self.alphabet.encode(seq_)
            seqs_padded_token.append(seq_token)
        return torch.tensor(seqs_padded_token)
        
    def mask_coord(self,coords, plddts):
        # for coords that plddt < coords_mask_plddt_th
        # coords -> inf & plddt -> -1
        # input: (torch.Size([8, 1093, 3, 3]), torch.Size([8, 1093]))
        # return: input same size
        bad_coords = plddts<self.coords_mask_plddt_th # torch.Size([8, 1093])
        plddts_ = torch.where(bad_coords, -1, plddts)
        bad_coords_unsqueezed = bad_coords.unsqueeze(-1).unsqueeze(-1).repeat(1,1,3, 3) # torch.Size([8, 1093, 3, 3])
        coords_ = torch.where(bad_coords_unsqueezed, np.inf, coords)
        return coords_, plddts_
            
    def __call__(self, batch):
        batch_size = len(batch)
        proteins = [i['protein'] for i in batch]
        #true_gos = [i['true_go'] for i in batch]
        #labels = [self.alphabet_go.tolabel(i['true_go']) for i in batch]
        #labels = torch.cat(labels, 0)
        #if self.device is not None:
        #    labels = labels.to(self.device)
        coords = [i['coords'] for i in batch]
        plddts = [i['plddt']/100 for i in batch]
        seqs = [i['seq'] for i in batch] #         seqs = self.batch_seq_converter(seqs)
        coords, plddts, seqs, tokens, padding_mask =\
            self.batch_coord_converter.from_lists(coords, plddts, seqs, self.device)
        coords,plddts = self.mask_coord(coords,plddts)
        seqs = self.batch_seq_converter(seqs)
        if self.device is not None:
            seqs = seqs.to(self.device)
        esm1vs = [i['esm1v'] for i in batch]
        esm1vs = self.batch_esm_converter(esm1vs)
        #print(coords.dtype, plddts.dtype, padding_mask.dtype, esm1vs.dtype)
        batched_dic = {'coords':coords, 
                       'plddts': plddts, 
                       'seqs': seqs,
                       'tokens': tokens, 
                       'padding_mask': padding_mask, 
                       'esm1vs': esm1vs, 
                       'proteins': proteins} 
                       #'true_gos': true_gos, 
                       #'labels': labels
        return batched_dic
