import os
import torch
import numpy as np
from torch.utils import data
from torch_geometric.data import Data
import pandas as pd
import json


class ELEA_Gaze_HeadPose_Processed(data.Dataset):
    def __init__(self, split_id, feature_path, num_segments, seg_len):
        super(ELEA_Gaze_HeadPose_Processed, self).__init__()
        self.split_id = split_id
        f = open('feat/ELEA_NumSeg16_SegLen16_MeanDiff.json')
        self.info_lb = json.load(f)
        self.selection = np.array(np.meshgrid(np.sort(np.asarray(split_id)), np.arange(num_segments))).T.reshape(-1,2)

        speak_f = open('feat/ELEA_SpeakingInterruption.json')
        self.speak_info = json.load(speak_f)

        f =  open('feat/ELEA_pagrankFt.json')
        self.pagrank_ft = json.load(f)

        self.hidden = 876

    def __getitem__(self, index):
        # No permutation
        vid = self.selection[index,0]
        seg = self.selection[index,1]

        target_set = self.info_lb[str(vid)]['DomRank']
        gaze_wt = self.info_lb[str(vid)]['vfoa_sum_wt']

        # Features - ---
        group_au_mean = self.info_lb[str(vid)]['Seg_'+str(seg)]['AUMean']
        group_au_difmean = self.info_lb[str(vid)]['Seg_'+str(seg)]['AUDiffMean']
        group_gaze = self.info_lb[str(vid)]['Seg_'+str(seg)]['GazeDiff']
        group_head = self.info_lb[str(vid)]['Seg_'+str(seg)]['HeadDiff']
        
        group_au_mean = torch.tensor(np.asarray(group_au_mean), dtype=torch.float)
        group_au_difmean = torch.tensor(np.asarray(group_au_difmean), dtype=torch.float)
        group_gaze = torch.tensor(np.asarray(group_gaze), dtype=torch.float)
        group_head = torch.tensor(np.asarray(group_head), dtype=torch.float)
        batch_edges = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3], # Defining fully connected graph
                           [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3]], dtype=torch.long)

        # Labels------

        target_set = np.asarray(target_set, dtype='int')

        # Speaking weight ----
        sp_wt = self.speak_info[str(vid)]['P2PInterruptions_FullVid']
        sp_wt = np.asarray(sp_wt, dtype = 'int')

        # normalising edge weights as gaze wt has higher values than speaking weights
        sp_wt = (sp_wt - np.min(sp_wt)) / (np.max(sp_wt) - np.min(sp_wt))
        gaze_wt = (gaze_wt - np.min(gaze_wt)) / (np.max(gaze_wt) - np.min(gaze_wt))


        if 100 in target_set: # replacing missing person as 0 label (lowest dominance rank)
            target_set[target_set == 100] = 0
            # change the highest rank to 3 in case of missing person is present. Missing person is 0 only

        # MFCC ft
        mft1 = np.asarray(self.speak_info[str(vid)]['P1_mfcc_4min'])
        mft2 = np.asarray(self.speak_info[str(vid)]['P2_mfcc_4min'])
        mft3 = np.asarray(self.speak_info[str(vid)]['P3_mfcc_4min'])
        mft4 = np.asarray(self.speak_info[str(vid)]['P4_mfcc_4min'])

        group_mfcc = np.zeros([4,96961]) # 96961 is the max len of all mfcc features of persons in all conv
        f = np.mean(mft1, axis = 0)
        group_mfcc[0:] = np.pad(f, (0, 96961 - len(f)), 'constant')
        f = np.mean(mft2, axis = 0)
        group_mfcc[1:] = np.pad(f, (0, 96961 - len(f)), 'constant')
        f = np.mean(mft3, axis = 0)
        group_mfcc[2:] = np.pad(f, (0, 96961 - len(f)), 'constant')
        f = np.mean(mft4, axis = 0)
        group_mfcc[3:] = np.pad(f, (0, 96961 - len(f)), 'constant')
        group_mfcc = torch.tensor(np.asarray(group_mfcc), dtype=torch.float)


        pg_p1 = self.pagrank_ft[str(vid)]['P1'] # PageRank features are the dominance features used in the baseline. Check paper.
        pg_p2 = self.pagrank_ft[str(vid)]['P2']
        pg_p3 = self.pagrank_ft[str(vid)]['P3']
        pg_p4 = self.pagrank_ft[str(vid)]['P4']

        ls_ll_ft = np.zeros([4,876])
        ls_ll_ft[0:] = np.pad(pg_p1, (0, 876 - len(pg_p1)), 'constant')
        ls_ll_ft[1:] = np.pad(pg_p2, (0, 876 - len(pg_p2)), 'constant')
        ls_ll_ft[2:] = np.pad(pg_p3, (0, 876 - len(pg_p3)), 'constant')
        ls_ll_ft[3:] = np.pad(pg_p4, (0, 876 - len(pg_p4)), 'constant')
        ls_ll_ft = torch.tensor(np.asarray(ls_ll_ft), dtype=torch.float)


        return Data(x = [group_au_mean, group_au_difmean,group_head, group_gaze, group_mfcc], edge_index=batch_edges, edge_weight = [torch.tensor(gaze_wt, dtype=torch.float), torch.tensor(sp_wt, dtype=torch.float)], y=torch.tensor(target_set, dtype=torch.float))

    def __len__(self):
        return len(self.selection) # selection, selection_perm



class ELEA_Baseline(data.Dataset):
    def __init__(self, split_id):
        super(ELEA_Baseline, self).__init__()
        self.split_id = split_id
        feature_file = open('feat/ELEA_pagrankFt.json')
        self.pagrank_ft = json.load(feature_file)
        self.labels= ELEA_labels(split_id, 'Perceived')
        self.hidden = 876
        self.p_hidd = 256

        f = open('feat/ELEA_NumSeg16_SegLen16_MeanDiff.json')
        self.info_lb = json.load(f)

    def __getitem__(self, index):
        vid_name = self.split_id[index]
        pg_p1 = np.asarray(self.pagrank_ft[str(vid_name)]['P1'])
        pg_p2 = np.asarray(self.pagrank_ft[str(vid_name)]['P2'])
        pg_p3 = np.asarray(self.pagrank_ft[str(vid_name)]['P3'])
        pg_p4 = np.asarray(self.pagrank_ft[str(vid_name)]['P4'])
        # print(len(pg_p1))
        if len(pg_p1)< self.hidden:
            pg_p1 = np.pad(pg_p1, (0, self.hidden - len(pg_p1)), 'constant')
        if len(pg_p2)< self.hidden:
            pg_p2 = np.pad(pg_p2, (0, self.hidden - len(pg_p2)), 'constant')
        if len(pg_p3)< self.hidden:
            pg_p3 = np.pad(pg_p3, (0, self.hidden - len(pg_p3)), 'constant')
        if len(pg_p4)< self.hidden:
            pg_p4 = np.pad(pg_p4, (0, self.hidden - len(pg_p4)), 'constant')


        pg_p1 = torch.tensor(np.asarray(pg_p1), dtype=torch.float)
        pg_p2 = torch.tensor(np.asarray(pg_p2), dtype=torch.float)
        pg_p3 = torch.tensor(np.asarray(pg_p3), dtype=torch.float)
        pg_p4 = torch.tensor(np.asarray(pg_p4), dtype=torch.float)
        ft = torch.stack((pg_p1, pg_p2,pg_p3,pg_p4))
        target_set = self.labels[self.labels[:, 0] == vid_name, 1:][0]
        if 100 in target_set: # replacing missing person as 0 label (lowest dominance rank)
            target_set[target_set == 100] = 0
        lb = np.zeros([4])
        lb[target_set==np.max(target_set)] = 1
        lb = np.expand_dims(lb, 1)

        gaze_wt = self.info_lb[str(vid_name)]['vfoa_sum_wt']


        batch_edges = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3],
                           [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3]], dtype=torch.long)
        

        return Data(x = ft, edge_index=batch_edges,edge_weight = torch.tensor(gaze_wt, dtype=torch.float), y=torch.tensor(lb, dtype=torch.long))

    def __len__(self):
        return len(self.split_id)

