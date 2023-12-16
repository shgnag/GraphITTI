import os
import torch
import numpy as np
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.metrics import average_precision_score, accuracy_score, confusion_matrix
from sklearn.metrics import mean_squared_error
import csv
import torch.nn.functional as F
import torch.nn as nn
from util import ranking_correlation, SPC_ranking_correlation


loss_ce = nn.CrossEntropyLoss()#nn.BCEWithLogitsLoss()#nn.CrossEntropyLoss()
softce = nn.NLLLoss()
bce_loss = nn.BCEWithLogitsLoss()

def optimize_graph(model, space_conf, dataloader_train,
                  data_loader_val, device, criterion, optimizer,
                  scheduler,num_seg, batch,num_epochs, models_out=None):
    best_acc = 0
    best_conf = None
    best_loss = 1000

    for epoch in range(num_epochs):
        print()
        print('\nEpoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        loss = _train_rank(model,space_conf, dataloader_train, optimizer, criterion, device,batch)
        _test_rank(model, space_conf, data_loader_val, criterion, device, batch, num_seg)
        # val_loss, val_ap, val_ap_same, val_ap_mid, acc, conf = _test_rank(model, space_conf, data_loader_val, criterion, device, batch)
    

        if(best_loss > loss):
            best_loss = min(loss, best_loss)
            model_target = os.path.join(models_out, str(best_loss)+'.pth')
            print('save model to ', model_target)

        scheduler.step()
        
    return model

def _train_rank(model,space_conf, dataloader, optimizer, criterion, device, batch):
    model.train()
    speakers, time_l = space_conf
    running_loss = 0.0
    list_kt = []
    # Iterate over data
    for idx, dl in enumerate(dataloader):

        print('\t Train iter {:d}/{:d} {:.4f}'.format(idx, len(dataloader), running_loss/(idx+1)) , end='\r')

        graph_data = dl
        graph_data = graph_data.to(device)
        targets = graph_data.y
        targets = targets.type(torch.LongTensor) 
        targets = targets.to(device)

        optimizer.zero_grad()
        with torch.set_grad_enabled(True):

            outputs, out_mdp  = model(graph_data)

            targets_new = targets.reshape([int(len(targets)/speakers),speakers])
            outputs_new = outputs.reshape([int(len(outputs)/speakers),speakers])
            # loss_rank = criterion(outputs_new, targets_new, device) #ranking

            # MDP loss
            out_mdp_new = out_mdp.reshape([int(len(out_mdp)/speakers),speakers])
            true_dom_id = torch.argmax(targets_new, dim = 1)
            loss_mdp = loss_ce(out_mdp_new, true_dom_id) #.float() with BCE

            kt = ranking_correlation(outputs_new,targets_new)
            list_kt.extend(kt)

            loss = loss_rank +loss_mdp
            
            loss.backward()
            optimizer.step()

        # statistics
        running_loss += loss.item()

    epoch_loss = running_loss / len(dataloader)

    print('Train Loss: {:.4f}'.format(epoch_loss),np.mean(np.array(list_kt)))

    return epoch_loss

def _test_rank(model, space_conf, dataloader, criterion, device, batch, num_seg):
    model.eval()  # Set model to evaluate mode

    speakers, time_l = space_conf
    mid = offset + (speakers+1)*int(time_l/2)

    y_true = []
    y_pred = []
    list_kt = list()
    list_sp = list()
    y_pred_vid = []
    y_true_vid = []
    # Iterate over data.
    for idx, dl in enumerate(dataloader):
        print('\t Val iter {:d}/{:d}'.format(idx, len(dataloader)) , end='\r')
        graph_data = dl
        graph_data = graph_data.to(device)
        targets = torch.flatten(graph_data.y)

        # forward
        with torch.set_grad_enabled(False):
            outputs, out_mdp = model(graph_data) 
            targets = targets.type(torch.LongTensor) 
            targets = targets.to(device)
            
            targets_new = targets.reshape([int(len(targets)/speakers),speakers])
            outputs_new = outputs.reshape([int(len(outputs)/speakers),speakers])
            # print('op', outputs_new)
            kt = ranking_correlation(outputs_new,targets_new)
            
            list_kt.extend(kt)
            sp = SPC_ranking_correlation(outputs_new,targets_new)
            list_sp.extend(sp)
            out_mdp_new = out_mdp.reshape([int(len(out_mdp)/speakers),speakers])

            true_dom_id = torch.argmax(targets_new, dim = 1)
            true_dom_id = true_dom_id.cpu().detach().numpy()
            pred_dom_id = torch.argmax(out_mdp_new, dim = 1)
            pred_dom_id = pred_dom_id.cpu().detach().numpy()
            y_true.extend(true_dom_id)
            y_pred.extend(pred_dom_id)


            y_pred_vid.append(np.argmax(np.bincount(pred_dom_id)))
            y_true_vid.append(np.argmax(np.bincount(true_dom_id)))


    sor = (list_sp - np.min(list_sp)) / (np.max(list_sp) - np.min(list_sp))
    skt = (list_kt - np.min(list_kt)) / (np.max(list_kt) - np.min(list_kt))

    acc = accuracy_score(y_true, y_pred)
    print('MDP segment wise and video wise Acc on VAL', acc, accuracy_score(y_true_vid, y_pred_vid))
    print(f" Average KT score on VAL: {np.mean(np.array(list_kt))} and std: {np.std(np.array(list_kt))}")
    print(f" Average SP score on VAL: {np.mean(np.array(list_sp))} and std: {np.std(np.array(list_sp))}")
    print(f" Average norm KT score on VAL: {np.mean(np.array(skt))} and std: {np.std(np.array(skt))}")
    print(f" Average norm SP score on VAL: {np.mean(np.array(sor))} and std: {np.std(np.array(sor))}")
    
    

def _evaluate_video(model, space_conf, dataloader, criterion, device):

    model.eval()  # Set model to evaluate mode

    speakers, num_seg = space_conf
    y_true_seg = []
    y_pred_seg = []
    y_true_vid = []
    y_pred_vid = []
    list_kt_seg = list()
    list_kt_vid = list()
    # Iterate over data.
    for idx, dl in enumerate(dataloader):
        print('\t Val iter {:d}/{:d}'.format(idx, len(dataloader)) , end='\r')
        graph_data = dl

        graph_data = graph_data.to(device)
        targets = torch.flatten(graph_data.y)

        # forward
        with torch.set_grad_enabled(False):
            outputs,out_mdp = model(graph_data)
            targets = targets.type(torch.LongTensor) 
            targets = targets.to(device)
            targets_new = targets.reshape([int(len(targets)/speakers),speakers])
            outputs_new = outputs.reshape([int(len(outputs)/speakers),speakers])


            kt = ranking_correlation(outputs_new,targets_new)

            list_kt_seg.extend(kt)

            list_kt_vid.append(np.mean(np.asarray(kt)))

            true_dom_id = torch.argmax(targets_new, dim = 1)
            true_dom_id = true_dom_id.cpu().detach().numpy()

            out_mdp_new = out_mdp.reshape([int(len(out_mdp)/speakers),speakers])
            pred_dom_id = torch.argmax(out_mdp_new, dim = 1)
            pred_dom_id = pred_dom_id.cpu().detach().numpy()

            y_true_seg.extend(true_dom_id)
            y_pred_seg.extend(pred_dom_id)

            y_pred_vid.append(np.argmax(np.bincount(pred_dom_id)))
            y_true_vid.append(np.argmax(np.bincount(true_dom_id)))
            # print('vid true',np.argmax(np.bincount(true_dom_id)))
            # print('vid pred',np.argmax(np.bincount(pred_dom_id)))

    print('Segment wise result--')
    acc = accuracy_score(y_true_seg, y_pred_seg)
    cf_matrix = list(confusion_matrix(y_true_seg, y_pred_seg))
    
    print(f" Average KT score on test graphs is: {np.mean(np.array(list_kt_seg))} and std: {np.std(np.array(list_kt_seg))}")
    print('MDP segment wise Acc', acc)
    print('MDP Confusion matrix', cf_matrix)

    print('\n\n Video wise result')
    acc = accuracy_score(y_true_vid, y_pred_vid)
    cf_matrix = list(confusion_matrix(y_true_vid, y_pred_vid))
    print('MDP segment wise Acc', acc)
    print(f" Average KT score on test graphs is: {np.mean(np.array(list_kt_vid))} and std: {np.std(np.array(list_kt_vid))}")
    print('MDP Confusion matrix', cf_matrix)

