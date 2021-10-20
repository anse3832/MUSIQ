import os

import torch
from tqdm import tqdm
import numpy as np
from scipy.stats import spearmanr, pearsonr


""" train model """
def train_epoch(config, epoch, model_transformer, model_backbone, criterion, optimizer, scheduler, train_loader):
    losses = []
    model_transformer.train()
    model_backbone.train()

    # input mask (batch_size x len_sqe+1)
    mask_inputs = torch.ones(config.batch_size, config.n_enc_seq+1).to(config.device)

    # save data for one epoch
    pred_epoch = []
    labels_epoch = []
    
    for data in tqdm(train_loader):
        # labels: batch size 
        # d_img_org: 3 x 768 x 1024
        # d_img_scale_1: 3 x 288 x 384
        # d_img_scale_2: 3 x 160 x 224
        d_img_org = data['d_img_org'].to(config.device)
        d_img_scale_1 = data['d_img_scale_1'].to(config.device)
        d_img_scale_2 = data['d_img_scale_2'].to(config.device)

        labels = data['score']
        labels = torch.squeeze(labels.type(torch.FloatTensor)).to(config.device)

        # backbone feature map (dis)
        # feat_dis_org: 2048 x 24 x 32
        # feat_dis_scale_1: 2048 x 9 x 12
        # feat_dis_scale_2: 2048 x 5 x 7
        feat_dis_org = model_backbone(d_img_org)
        feat_dis_scale_1 = model_backbone(d_img_scale_1)
        feat_dis_scale_2 = model_backbone(d_img_scale_2)

        # this value should be extracted from backbone network
        # enc_inputs_embed: batch x len_seq x n_feat
        

        # weight update
        optimizer.zero_grad()

        pred = model_transformer(mask_inputs, feat_dis_org, feat_dis_scale_1, feat_dis_scale_2)
        loss = criterion(torch.squeeze(pred), labels)
        loss_val = loss.item()
        losses.append(loss_val)

        loss.backward()
        optimizer.step()
        scheduler.step()

        # save results in one epoch
        pred_batch_numpy = pred.data.cpu().numpy()
        labels_batch_numpy = labels.data.cpu().numpy()
        pred_epoch = np.append(pred_epoch, pred_batch_numpy)
        labels_epoch = np.append(labels_epoch, labels_batch_numpy)
    
    
    # compute correlation coefficient
    rho_s, _ = spearmanr(np.squeeze(pred_epoch), np.squeeze(labels_epoch))
    rho_p, _ = pearsonr(np.squeeze(pred_epoch), np.squeeze(labels_epoch))

    print('[train] epoch:%d / loss:%f / SROCC:%4f / PLCC:%4f' % (epoch+1, loss.item(), rho_s, rho_p))

    # save weights
    if (epoch+1) % config.save_freq == 0:
        weights_file_name = "epoch%d.pth" % (epoch+1)
        weights_file = os.path.join(config.snap_path, weights_file_name)
        torch.save({
            'epoch': epoch,
            'model_backbone_state_dict': model_backbone.state_dict(),
            'model_transformer_state_dict': model_transformer.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': loss
        }, weights_file)
        print('save weights of epoch %d' % (epoch+1))

    return np.mean(losses), rho_s, rho_p


""" validation """
def eval_epoch(config, epoch, model_transformer, model_backbone, criterion, test_loader):
    with torch.no_grad():
        losses = []
        model_transformer.eval()
        model_backbone.eval()

        # value is not changed
        mask_inputs = torch.ones(config.batch_size, config.n_enc_seq+1).to(config.device)

        # save data for one epoch
        pred_epoch = []
        labels_epoch = []

        for data in tqdm(test_loader):
            # labels: batch size 
            # d_img_org: batch x 3 x 768 x 1024
            # d_img_scale_1: batch x 3 x 288 x 384
            # d_img_scale_2: batch x 3 x 160 x 224

            d_img_org = data['d_img_org'].to(config.device)
            d_img_scale_1 = data['d_img_scale_1'].to(config.device)
            d_img_scale_2 = data['d_img_scale_2'].to(config.device)

            labels = data['score']
            labels = torch.squeeze(labels.type(torch.FloatTensor)).to(config.device)

            # backbone featuremap
            # feat_dis_org: batch x 2048 x 24 x 32
            # feat_dis_scale_1: batch x 2048 x 9 x 12
            # feat_dis_scale_2: batch x 2048 x 5 x 12
            feat_dis_org = model_backbone(d_img_org)
            feat_dis_scale_1 = model_backbone(d_img_scale_1)
            feat_dis_scale_2 = model_backbone(d_img_scale_2)

            pred = model_transformer(mask_inputs, feat_dis_org, feat_dis_scale_1, feat_dis_scale_2)            

            # compute loss
            loss = criterion(torch.squeeze(pred), labels)
            loss_val = loss.item()
            losses.append(loss_val)

            # save results in one epoch
            pred_batch_numpy = pred.data.cpu().numpy()
            labels_batch_numpy = labels.data.cpu().numpy()
            pred_epoch = np.append(pred_epoch, pred_batch_numpy)
            labels_epoch = np.append(labels_epoch, labels_batch_numpy)
        
        # compute correlation coefficient
        rho_s, _ = spearmanr(np.squeeze(pred_epoch), np.squeeze(labels_epoch))
        rho_p, _ = pearsonr(np.squeeze(pred_epoch), np.squeeze(labels_epoch))

        print('test epoch:%d / loss:%f /SROCC:%4f / PLCC:%4f' % (epoch+1, loss.item(), rho_s, rho_p))

        return np.mean(losses), rho_s, rho_p
