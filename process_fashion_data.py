import h5py
from scipy.io import loadmat
import cv2
import numpy as np
from torch.utils.serialization import load_lua
import os
import torch
import cPickle
import random
import shutil

root = './datasets'
alphabet = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{} "

with h5py.File('./G2.h5', 'r') as f:
    a = loadmat('./language_original.mat')
    # dict_keys(['__header__', '__version__', '__globals__', 'engJ', 'cate_new', 'color_', 'gender_', 'sleeve_', 'img_root', 'nameList', 'codeJ'])
    index = loadmat('./ind.mat')
    # ['__globals__', '__header__', 'test_ind', 'test_set_pair_ind', 'train_ind', '__version__']
    for i in range(70000):
        dict = {}
        init = np.zeros((104,1))
        count = 0
        for char in a['engJ'][index['train_ind'][i,0]-1][0][0]:
            init[count, 0] = alphabet.index(char) + 1
            count += 1
        init = torch.ByteTensor(init)
        dict['char'] = init
        dict['img'] = a['nameList'][index['train_ind'][i,0]-1][0][0]
        name = a['nameList'][index['train_ind'][i, 0] - 1][0][0]
        if not os.path.exists(os.path.join(root, 'FashionGAN_txt', name.strip().split('/')[1])):
            os.makedirs(os.path.join(root, 'FashionGAN_txt', name.strip().split('/')[1]))
        with open(os.path.join(root, 'FashionGAN_txt', name.strip().split('/')[1], name.strip().split('/')[-1][:-4] + '.pkl'), 'wb') as pkl_file:
            cPickle.dump(dict, pkl_file)

        print(type(f['ih'][0]))
        print(f['ih_mean'])
        img = f['ih'][index['train_ind'][i,0]-1].transpose(2,1,0) + f['ih_mean'].value.transpose(2,1,0)
        if not os.path.exists(os.path.join(root, 'img', name.strip().split('/')[1])):
            os.makedirs(os.path.join(root,'img',  name.strip().split('/')[1]))

        if os.path.isfile(os.path.join(root, name)):
            continue
        else:
            cv2.imwrite(os.path.join(root, name), np.stack([img[:,:,2],img[:,:,1],img[:,:,0]],axis=-1) * 255.)

with open(os.path.join(root, 'FashionGAN_txt','trainclasses.txt'), 'wb') as f:
    for _, dirs, file in os.walk('./datasets/img/'):
        for dir in dirs:
            f.write(dir + '\n')
