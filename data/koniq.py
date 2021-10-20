import os
import torch
import numpy as np
import cv2

class IQADataset(torch.utils.data.Dataset):
    def __init__(self, db_path, txt_file_name, scale_1, scale_2, transform, train_mode, scene_list, train_size=0.8):
        super(IQADataset, self).__init__()

        self.db_path = db_path
        self.txt_file_name = txt_file_name
        self.scale_1 = scale_1
        self.scale_2 = scale_2
        self.transform = transform
        self.train_mode = train_mode
        self.scene_list = scene_list
        self.train_size = train_size

        self.data_dict = IQADatalist(
            txt_file_name = self.txt_file_name,
            train_mode = self.train_mode,
            scene_list = self.scene_list,
            train_size = self.train_size
        ).load_data_dict()

        self.n_images = len(self.data_dict['d_img_list'])
    
    def __len__(self):
        return self.n_images
    
    def __getitem__(self, idx):
        # d_img_org: H x W x C
        d_img_name = self.data_dict['d_img_list'][idx]
        d_img_org = cv2.imread(os.path.join((self.db_path + '/1024x768'), d_img_name), cv2.IMREAD_COLOR)
        d_img_org = cv2.cvtColor(d_img_org, cv2.COLOR_BGR2RGB)
        d_img_org = np.array(d_img_org).astype('float32') / 255

        h, w, c = d_img_org.shape
        d_img_scale_1 = cv2.resize(d_img_org, dsize=(self.scale_1, int(h*(self.scale_1/w))), interpolation=cv2.INTER_CUBIC)
        d_img_scale_2 = cv2.resize(d_img_org, dsize=(self.scale_2, int(h*(self.scale_2/w))), interpolation=cv2.INTER_CUBIC)
        d_img_scale_2 = d_img_scale_2[:160, :, :]

        score = self.data_dict['score_list'][idx]

        sample = {'d_img_org': d_img_org, 'd_img_scale_1': d_img_scale_1, 'd_img_scale_2':d_img_scale_2, 'score': score}

        if self.transform:
            sample = self.transform(sample)
        
        return sample


class IQADatalist():
        def __init__(self, txt_file_name, train_mode, scene_list, train_size=0.8):
            self.txt_file_name = txt_file_name
            self.train_mode = train_mode
            self.train_size = train_size
            self.scene_list = scene_list
        
        def load_data_dict(self):
            scn_idx_list, d_img_list, score_list = [], [], []

            # list append
            with open(self.txt_file_name, 'r') as listFile:
                for line in listFile:
                    scn_idx, dis, score = line.split()
                    scn_idx = int(scn_idx)
                    score = float(score)

                    scene_list = self.scene_list

                    # add items according to scene number
                    if scn_idx in scene_list:
                        scn_idx_list.append(scn_idx)
                        d_img_list.append(dis)
                        score_list.append(score)
                
            # reshape score_list (1xn -> nx1)
            score_list = np.array(score_list)
            score_list = score_list.astype('float').reshape(-1, 1)

            data_dict = {'d_img_list': d_img_list, 'score_list': score_list}
            
            return data_dict
