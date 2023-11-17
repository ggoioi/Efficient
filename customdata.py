import glob

import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import os
import random
from PIL import Image, UnidentifiedImageError, ImageFile

class CustomImageDataset(Dataset):
    def __init__(self, files, labels, class_to_idx, transform):
        super(CustomImageDataset, self).__init__()
        self.files = files
        self.labels = labels
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # file 경로
        file = self.files[idx]
        # PIL.Image로 이미지 로드
        img = Image.open(file).convert('RGB')
        # transform 적용
        img = self.transform(img)
        # label 생성
        lbl = self.class_to_idx[self.labels[idx]]
        # image, label return
        return img, lbl

if __name__ == '__main__':

    SEED = 1234

    train_images = []
    test_images = []
    valid_images = []
    folders = glob.glob('./Data/*')
    for folder in folders:
        test_file = []
        valid_file = []
        train_file = []
        temp = []
        labels = os.path.basename(folder)
        files = sorted(glob.glob(folder + '/*'))

        # 각 Label별 이미지 데이터셋 셔플
        random.seed(SEED)
        random.shuffle(files)
        for i in range(1331):
            test_file.append(files[i])
            # del files[i]
            # files.reset_index(inplace = True)
            temp.append(files[i])
        for j in range(1331, 2662):
            valid_file.append(files[j])
            temp.append(files[j])
            # del files[j]
        temp = list(set(files) - set(temp))
        if temp is not None:
            train_file = temp
        test_images.extend(test_file)
        valid_images.extend(valid_file)
        train_images.extend(train_file)
    random.shuffle(test_images)
    random.shuffle(valid_images)
    random.shuffle(train_images)
    class_to_idx = {os.path.basename(f): idx for idx, f in enumerate(folders)}

    valid_labels = [f.split("\\")[1] for f in valid_images]
    test_labels = [f.split("\\")[1] for f in test_images]
    train_labels = [f.split("\\")[1] for f in train_images]
    print('===' * 10)
    print(f'train images: {len(train_images)}')
    print(f'train labels: {len(train_labels)}')
    print(f'test images: {len(test_images)}')
    print(f'test labels: {len(test_labels)}')
    print(f'valid images: {len(valid_images)}')
    print(f'valid labels: {len(valid_labels)}')
    # print(valid_labels)
    if 'Bad' in train_labels:
        print('실패')
    else:
        print('성공')

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    #     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), # 이미지 정규화
    ])

    valid_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    #     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), # 이미지 정규화
    ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    #     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), # 이미지 정규화
    ])

    train_dataset = CustomImageDataset(train_images, train_labels, class_to_idx, train_transform)
    valid_dataset = CustomImageDataset(valid_images, valid_labels, class_to_idx, valid_transform)
    test_dataset = CustomImageDataset(test_images, test_labels, class_to_idx, test_transform)

    train_loader = DataLoader(train_dataset,
                              batch_size=32,
                              shuffle=True,
                              num_workers=8
                             )

    valid_loader = DataLoader(valid_dataset,
                              batch_size=32,
                              shuffle=True,
                              num_workers=8
                             )

    test_loader = DataLoader(test_dataset,
                             batch_size=32,
                             shuffle=True,
                             num_workers=8
                            )

    images, labels = next(iter(valid_loader))

    print(images[0].shape)

    # ImageFolder의 속성 값인 class_to_idx를 할당
    labels_map = {v: k for k, v in train_dataset.class_to_idx.items()}

    figure = plt.figure(figsize=(12, 8))
    cols, rows = 8, 4

    # 이미지를 출력합니다. RGB 이미지로 구성되어 있습니다.
    for i in range(1, cols * rows + 1):
        sample_idx = torch.randint(len(images), size=(1,)).item()
        img, label = images[sample_idx], labels[sample_idx].item()
        figure.add_subplot(rows, cols, i)
        plt.title(labels_map[label])
        plt.axis("off")
        # 본래 이미지의 shape은 (3, 300, 300) 입니다.
        # 이를 imshow() 함수로 이미지 시각화 하기 위하여 (300, 300, 3)으로 shape 변경을 한 후 시각화합니다.
        plt.imshow(torch.permute(img, (1, 2, 0)))
    plt.show()