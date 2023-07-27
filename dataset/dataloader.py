# pylint: skip-file
import os
import pandas as pd
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset
import torch
from torch.nn.utils.rnn import pad_sequence
import cv2
import numpy as np


class BatchPad:
    def __call__(self, batch):
        image_seq = pad_sequence([item[0] for item in batch], batch_first=True)
        targets = torch.tensor([item[1] for item in batch])
        return image_seq, targets
    

class BatchPadV2:
    def __call__(self, batch):
        image_seq = pad_sequence([item[0] for item in batch], batch_first=True)
        targets = torch.tensor([item[1] for item in batch])
        actual_lens = torch.tensor([item[2] for item in batch])
        return image_seq, targets, actual_lens
    

class __ImageDataset(Dataset):
    def __init__(self,
                 dataset_status:str,
                 csv_path:str,
                 data_dir:str='/home/public_datasets/FERV39k'):
        """
        dataset instance of FERV39k image level data
        Args: 
            dataset_status: 'train' or 'test'
            csv_path: path of csv file
            data_dir: directory of dataset
        """
        super(__ImageDataset, self).__init__()
        
        # process info_csv file to get video_df and label_df
        self.dataset_status = dataset_status 
        info_df = pd.read_csv(csv_path, header=None)
        info_df[[0, 1]] = info_df[0].str.split(' ', expand=True)
        labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        info_df = pd.get_dummies(info_df, columns=[1])
        info_df = info_df.rename(columns={'1_Angry': 'Angry', 
                                          '1_Disgust': 'Disgust', 
                                          '1_Fear':'Fear', 
                                          '1_Happy':'Happy', 
                                          '1_Neutral':'Neutral',	
                                          '1_Sad':'Sad',
                                          '1_Surprise':'Surprise'})
        self.video_df = info_df[[0]]
        label_df = info_df[labels]
        self.label_df = np.argmax(label_df.values, axis=1)

        # load parameters
        self.data_dir = data_dir

        # transforms
        self.transforms = torch.nn.Sequential(
            transforms.RandomHorizontalFlip(0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.08)
        )

    def __getitem__(self, index):
        """
        Args:
            index: the index of item
        Returns:
            input data and its labels of item 
        """
        video_id = self.video_df.iloc[index, 0]
        label = self.label_df[index]
        data, actual_len = self.get_data_from_video_id(video_id)
        # data = self.transform(data)
        return data, label, actual_len

    def __len__(self):
        return len(self.video_df)

    def get_data_from_video_id(self, video_id:str):
        datapath = self.data_dir + '/2_ClipsforFaceCrop/' + video_id

        filenames = sorted(os.listdir(datapath))
        #data = [torchvision.io.read_image(datapath + '/' + filename) for filename in filenames]
        data = [torch.from_numpy(cv2.cvtColor(cv2.imread(datapath + '/' + filename), cv2.COLOR_BGR2RGB)).permute(2,0,1) for filename in filenames]
        transformed = []
        if self.dataset_status == 'train':
            state = torch.get_rng_state()
            for img in data:
                transformed.append(self.transforms(img))
                torch.set_rng_state(state)
            return torch.stack(transformed) / 255.0, len(transformed)
        if self.dataset_status == 'test':
            return torch.stack(data) / 255.0, len(data)


def load_data(data_dir, csv_train, csv_test, args):
    """
    load data and sampler according to info list
    Args:
        data_dir (str): dataset directory
        csv_train (str): csv of train set
        csv_test (str): csv of test set
        args (parser): _description_

    Returns:
        train_data (Dataset): train dataset
        test_data (Dataset): test dataset
        train_sampler (Sampler): train sampler
        test_sampler (Sampler): test sampler    
    """ 
    train_data = __ImageDataset('train', csv_train, data_dir)
    test_data = __ImageDataset('test', csv_test, data_dir)
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_data)
    else:
        train_sampler = torch.utils.data.RandomSampler(train_data)
        test_sampler = torch.utils.data.SequentialSampler(test_data)    
    return train_data, test_data, train_sampler, test_sampler
