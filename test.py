import os
import argparse
from fastText import FastText
from PIL import Image

import torch
from torch.autograd import Variable
import torchvision.transforms as transforms
from torchvision.utils import save_image

from model import VisualSemanticEmbedding
from model import Generator
from data import split_sentence_into_words
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('--img_root', type=str, required=True,
                    help='root directory that contains images')
parser.add_argument('--text_file', type=str, required=True,
                    help='text file that contains descriptions')
parser.add_argument('--fasttext_model', type=str, required=True,
                    help='pretrained fastText model (binary file)')
parser.add_argument('--text_embedding_model', type=str, required=True,
                    help='pretrained text embedding model')
parser.add_argument('--embed_ndim', type=int, default=300,
                    help='dimension of embedded vector (default: 300)')
parser.add_argument('--generator_model', type=str, required=True,
                    help='pretrained generator model')
parser.add_argument('--use_vgg', action='store_true',
                    help='use pretrained VGG network for image encoder')
parser.add_argument('--output_root', type=str, required=True,
                    help='root directory of output')
parser.add_argument('--no_cuda', action='store_true',
                    help='do not use cuda')
parser.add_argument('--fusing_method', type=str, default='',
                    help='fusing_method')
args = parser.parse_args()

if not args.no_cuda and not torch.cuda.is_available():
    print('Warning: cuda is not available on this machine.')
    args.no_cuda = True


if __name__ == '__main__':
    print('Loading a pretrained fastText model...')
    word_embedding = FastText.load_model(args.fasttext_model)

    print('Loading a pretrained model...')

    txt_encoder = VisualSemanticEmbedding(args.embed_ndim)
    txt_encoder.load_state_dict(torch.load(args.text_embedding_model))
    txt_encoder = txt_encoder.txt_encoder

    G = Generator(use_vgg=args.use_vgg, fusing=args.fusing_method)
    G.load_state_dict(torch.load(args.generator_model))
    G.train(False)

    if not args.no_cuda:
        txt_encoder.cuda()
        G.cuda()

    transform = transforms.Compose([
        transforms.Scale(70),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    transform2 = transforms.Compose([
        transforms.Scale(70),
        transforms.CenterCrop(64),
        transforms.ToTensor()
    ])

    print('Loading test data...')
    filenames = sorted(os.listdir(args.img_root))
    img = []
    img_save = []
    for fn in filenames:
        im = Image.open(os.path.join(args.img_root, fn))
        im_save = transform2(im)
        im = transform(im)
        img.append(im)
        img_save.append(im_save)
    img = torch.stack(img)
    img_save = torch.stack(img_save)
    save_image(img_save, os.path.join(args.output_root, 'original.jpg'),nrow=6, padding=0)
    img = Variable(img.cuda() if not args.no_cuda else img, volatile=True)

    html = '<html><body><h1>Manipulated Images</h1><table border="1px solid gray" style="width=100%"><tr><td><b>Description</b></td><td><b>Image</b></td></tr>'
    html += '\n<tr><td>ORIGINAL</td><td><img src="{}"></td></tr>'.format('original.jpg')
    with open(args.text_file, 'r') as f:
        texts = f.readlines()

    for i, txt in enumerate(texts):
        txt = txt.replace('\n', '')
        desc = split_sentence_into_words(txt)
        desc = torch.Tensor([word_embedding.get_word_vector(w).astype(np.float64) for w in desc])
        desc = desc.unsqueeze(1)
        desc = desc.repeat(1, img.size(0), 1)
        desc = Variable(desc.cuda() if not args.no_cuda else desc, volatile=True)

        _, txt_feat = txt_encoder(desc)
        txt_feat = txt_feat.squeeze(0)
        output, _ = G(img, txt_feat)

        out_filename = 'output_%d.jpg' % i
        save_image((output.data + 1) * 0.5, os.path.join(args.output_root, out_filename),nrow=6, padding=0)
        html += '\n<tr><td>{}</td><td><img src="{}"></td></tr>'.format(txt, out_filename)

    with open(os.path.join(args.output_root, 'index.html'), 'w') as f:
        f.write(html)
    print('Done. The results were saved in %s.' % args.output_root)
