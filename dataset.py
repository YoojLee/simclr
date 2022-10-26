import cv2
import os, glob

import torch
from torch.utils.data import Dataset

from augmentation import SimCLRTransform


class SimCLRDataset(Dataset):
    def __init__(self, data_root:str, is_train:bool=True, transforms = None, transforms_eval = None, label_info="label.txt", downsample=False):
        """
        Dataset Class for SimCLR. Should be unlableled.

        - Args
            data_root (str): a directory data stored.
            is_train (bool): indicates the instance will be for training dataset or not
            transforms (Transforms): the family of augmentations to be applied for a single image batch.
            label_info (str): a txt file containing label information.

        """
        super(SimCLRDataset, self).__init__()
        
        self.is_train = is_train
        self.transforms = transforms
        self.transforms_eval = transforms_eval
        self.label_names = []
        self._label_map = dict()
        
        if self.transforms == None:
            raise ValueError("Transforms must be given.")

        if self.is_train:
            self.data_root = os.path.join(data_root, "train")
        else:
            self.data_root = os.path.join(data_root, "val")

        # labels info
        with open(os.path.join(data_root, label_info), "r") as f:
            _labels = list(map(lambda x: x.strip().split(" "), f.readlines()))
        
        for cls, cls_n, cls_name in _labels:
            self._label_map[cls] = int(cls_n)-1 # label을 zero-index로 넣어주지 않으면 n_classes보다 큰 label이 들어왔다는 error를 리턴하게 됨.
            self.label_names.append(cls_name)

        self.img_list = glob.glob(f"{self.data_root}/**/*.JPEG", recursive=True) # image-net과 같은 경우에는 확장자가 JPEG only.
        self.labels = list(map(lambda x: self._label_map[x.split("/")[-2]], self.img_list))

        if downsample: # 1/10으로 downsampling
            self.img_list = self.img_list[::100]
            self.labels = self.labels[::100]
        

    def __len__(self):
        return len(self.img_list)

    
    def __getitem__(self, index: int):
        """
        transformation은 동일한 구성인데 거기서 probability가 어떻게 적용되느냐에 따라 다른 transformation이다 이러는 것 같음..아마?
        """
        img = cv2.imread(self.img_list[index]) # 이미지 경로 읽어오기 (H,W,C)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # color space 변환 & (C,H,W)로 변경

        if self.transforms:
            im1 = self.transforms(img)['image'] # albumentations 타입의 transform 적용
            im2 = self.transforms(img)['image']

        if self.transforms_eval:
            im_eval = self.transforms_eval(img)['image']

        label = self.labels[index]

        return torch.cat((im1, im2)), im_eval, label # 이 부분 다시 체크 (이걸 어떻게 하겠다는 건지 아직 잘 모르겠는걸..)

if __name__ == "__main__":
    data_root = "/home/data/ImageNet"
    is_train = True
    transforms = SimCLRTransform()
    dataset = SimCLRDataset(data_root, is_train, downsample=False, transforms=transforms)
    print(len(dataset))
    
    from torch.utils.data import DataLoader
    
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    print(next(iter(dataloader))[0].shape) # 이렇게 view 적용해주는 건 맞음.
    
    for idx, data in enumerate(dataloader):
        img, label = data
        print(img.shape)
        print(label.shape)
        print(label)
        print(label.flatten())
        if idx == 10:
            break

    # for i in range(0, 10, 2):
    #     tmp = next(iter(dataloader))[i//2].view(-1,3,224,224)
    #     cv2.imwrite(f"test{i}.jpg", tmp[0].permute(1,2,0).numpy())
    #     cv2.imwrite(f"test{i+1}.jpg", tmp[1].permute(1,2,0).numpy())
    