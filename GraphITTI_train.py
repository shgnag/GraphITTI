import sys
import torch

import config as exp_conf
import models_GraphITTI as model

from torch.optim import lr_scheduler
from torch_geometric.data import DataLoader

from datasets import *

from util import rank_loss_cal, preprocess_ELEA
from optimization import optimize_graph, _evaluate_video
import warnings
warnings.filterwarnings("ignore")


if __name__ == '__main__':
    # experiment Reproducibility
    torch.manual_seed(11)
    torch.cuda.manual_seed(22)
    torch.backends.cudnn.deterministic = True
    print(sys.argv)

    time_length = 1 # static
    speakers = 4 # num of nodes in one graph
    hidden_size = 64
    num_attr = 5

    num_segments = 16 # Total num of segments
    seg_len = 16 # num of frames in each sequence


    io_config = exp_conf.ELEA_inputs
    opt_config = exp_conf.ELEA_optimization_params

    # io config
    model_name = 'Model_GraphITTI_seg'+str(num_segments)+'_len_'+str(seg_len)#+'filters_'
    target_models = io_config['models_out']

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Using ', device)
    

    model = model.IntraInterAtt(num_attr,speakers, hidden_size, device)
    # model = model.GroupDNNIJCAI(num_attr,speakers, hidden_size)
    model.to(device)


    criterion = rank_loss_cal
    optimizer = opt_config['optimizer'](model.parameters(),lr=opt_config['learning_rate'])
    scheduler = lr_scheduler.StepLR(optimizer, step_size=opt_config['step_size'],
                                    gamma=opt_config['gamma'])


    # ELEA
    dataset_train = ELEA_Gaze_HeadPose_Processed(io_config['train_id'],io_config['features_path'], num_segments, seg_len)
    dataset_val = ELEA_Gaze_HeadPose_Processed(io_config['val_id'],io_config['features_path'], num_segments, seg_len)

    # dataset_train = ELEA_Baseline(io_config['train_id'])
    # dataset_val = ELEA_Baseline(io_config['val_id'])

    dl_train = DataLoader(dataset_train, batch_size=opt_config['batch_size'],
                          shuffle=True, num_workers=opt_config['threads'])
    dl_val = DataLoader(dataset_val, batch_size=num_segments,
                        shuffle=False, num_workers=opt_config['threads'])


    optimize_graph(model, (speakers, time_length), dl_train, dl_val, device,
                             criterion, optimizer, scheduler,num_segments, opt_config['batch_size'],
                             num_epochs=opt_config['epochs'],
                             models_out=target_models)

    
    # model_path = '/models/ELEA_GraphITTI_seg16_len_16.pth'
    # checkpoint = torch.load(model_path)
    # model.load_state_dict(checkpoint)
    # _evaluate_video(model, (1, speakers, num_segments), dl_test, criterion, device)
