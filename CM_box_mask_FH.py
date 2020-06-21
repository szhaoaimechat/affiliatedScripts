#!/usr/bin/python
#
# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This script can be used to sample many images from a model for evaluation.
"""


import argparse, json
import os, random, math
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix 

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision
from torchvision.transforms import ToTensor
import torch.nn.functional as F
from torchvision.utils import save_image

from scipy.misc import imsave, imresize

from sg2im.data import imagenet_deprocess_batch
from sg2im.data.coco import CocoSceneGraphDataset, coco_collate_fn
from sg2im.data.vg import VgSceneGraphDataset, vg_collate_fn
from sg2im.data.utils import split_graph_batch
from sg2im.model import Sg2ImModel
from sg2im.utils import int_tuple, bool_flag
from sg2im.vis import draw_scene_graph

import numpy as np
import pandas as pd
import seaborn as sns
from tqdm.notebook import tqdm


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from sklearn.preprocessing import MinMaxScaler    
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report


#from torch.utils.tensorboard import SummaryWriter
#from datetime import datetime
#now = datetime.now()
#current_time = now.strftime("%H_%M_%S")
#RUN_DIR = os.path.join('run_tensorboard', current_time)
#print(RUN_DIR)
#writer = SummaryWriter(RUN_DIR)


parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', default= None)#'sg2im-models/coco64.pt')
parser.add_argument('--checkpoint_list', default=None)
parser.add_argument('--model_mode', default='eval', choices=['train', 'eval'])

# Shared dataset options
parser.add_argument('--dataset', default='coco', choices=['coco', 'vg'])
parser.add_argument('--which_data', default='val', choices=['train', 'val'])

parser.add_argument('--image_size', default=(64, 64), type=int_tuple)
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--shuffle', default=False, type=bool_flag)
parser.add_argument('--loader_num_workers', default=4, type=int)
parser.add_argument('--num_samples', default=100000, type=int)
parser.add_argument('--save_gt_imgs', default=False, type=bool_flag)
parser.add_argument('--save_graphs', default=False, type=bool_flag)
parser.add_argument('--use_gt_boxes', default=False, type=bool_flag)
parser.add_argument('--use_gt_masks', default=False, type=bool_flag)
parser.add_argument('--save_layout', default=True, type=bool_flag)
parser.add_argument('--output_dir', default='output_CMs')



VG_DIR = os.path.expanduser('/home/dell/Documents/datasets/vg')
COCO_DIR = os.path.expanduser('/home/dell/Documents/datasets/coco')
# For VG
#VG_DIR = os.path.expanduser('../../datasets/vg')
parser.add_argument('--vg_h5', default=os.path.join(VG_DIR, 'val.h5'))
parser.add_argument('--vg_image_dir',
        default=os.path.join(VG_DIR, 'images'))

# For COCO


parser.add_argument('--coco_train_image_dir',
         default=os.path.join(COCO_DIR, 'images/train2017'))
parser.add_argument('--coco_val_image_dir',
         default=os.path.join(COCO_DIR, 'images/val2017'))
parser.add_argument('--coco_train_instances_json',
         default=os.path.join(COCO_DIR, 'annotations/instances_train2017.json'))
parser.add_argument('--coco_train_stuff_json',
         default=os.path.join(COCO_DIR, 'annotations/stuff_train2017.json'))
parser.add_argument('--coco_val_instances_json',
         default=os.path.join(COCO_DIR, 'annotations/instances_val2017.json'))
parser.add_argument('--coco_val_stuff_json',
         default=os.path.join(COCO_DIR, 'annotations/stuff_val2017.json'))
#parser.add_argument('--instance_whitelist', default=None, type=str_tuple)
#parser.add_argument('--stuff_whitelist', default=None, type=str_tuple)
parser.add_argument('--coco_include_other', default=False, type=bool_flag)
parser.add_argument('--min_object_size', default=0.02, type=float)
parser.add_argument('--min_objects_per_image', default=3, type=int)
parser.add_argument('--coco_stuff_only', default=True, type=bool_flag)

parser.add_argument('--coco_train_oldlist_txt',
                    default=os.path.join(COCO_DIR, 'annotations/deprecated-challenge2017/train-ids.txt'))
parser.add_argument('--coco_val_oldlist_txt',
                    default=os.path.join(COCO_DIR, 'annotations/deprecated-challenge2017/val-ids.txt'))

parser.add_argument('--FH_dir_train', default = 'dataFH/train_FH.npy')
parser.add_argument('--FH_dir_val', default = 'dataFH/val_FH.npy')

colors = [(250,128,114),(32,178,170),(240,128,128),(248,248,255),(175,238,238), (176,224,230)]

def build_coco_dset(args, checkpoint):

  checkpoint_args = checkpoint['args']
  dset_kwargs = {
    'image_dir': args.coco_train_image_dir,
    'instances_json': args.coco_train_instances_json,
    'stuff_json': args.coco_train_stuff_json,
    'stuff_only': checkpoint_args['coco_stuff_only'],
    'oldlist_txt':args.coco_train_oldlist_txt,     
    'image_size': args.image_size,
    'mask_size': checkpoint_args['mask_size'],
    'max_samples': args.num_samples,
    'min_object_size': checkpoint_args['min_object_size'],
    'min_objects_per_image': checkpoint_args['min_objects_per_image'],
    'instance_whitelist': checkpoint_args['instance_whitelist'],
    'stuff_whitelist': checkpoint_args['stuff_whitelist'],
    'include_other': checkpoint_args.get('coco_include_other', True),
  }
  # training set
  if(args.which_data == 'train'):
      dset = CocoSceneGraphDataset(**dset_kwargs)
  else:  # validation set
      dset_kwargs['image_dir'] = args.coco_val_image_dir
      dset_kwargs['instances_json'] = args.coco_val_instances_json
      dset_kwargs['stuff_json'] = args.coco_val_stuff_json
      dset_kwargs['oldlist_txt'] = args.coco_val_oldlist_txt
      dset = CocoSceneGraphDataset(**dset_kwargs)
    
  num_objs = dset.total_objects()
  num_imgs = len(dset)
  print('Training dataset has %d images and %d objects' % (num_imgs, num_objs))
  print('(%.2f objects per image)' % (float(num_objs) / num_imgs))
  print('Training dataset has %d images and %d objects' % (num_imgs, num_objs))
  print('(%.2f objects per image)' % (float(num_objs) / num_imgs))
  
  return dset


def build_vg_dset(args, checkpoint):
  vocab = checkpoint['model_kwargs']['vocab']
  dset_kwargs = {
    'vocab': vocab,
    'h5_path': args.vg_h5,
    'image_dir': args.vg_image_dir,
    'image_size': args.image_size,
    'max_samples': args.num_samples,
    'max_objects': checkpoint['args']['max_objects_per_image'],
    'use_orphaned_objects': checkpoint['args']['vg_use_orphaned_objects'],
  }
  dset = VgSceneGraphDataset(**dset_kwargs)
  return dset


def build_loader(args, checkpoint):
  if args.dataset == 'coco':
    dset = build_coco_dset(args, checkpoint)
    collate_fn = coco_collate_fn
  elif args.dataset == 'vg':
    dset = build_vg_dset(args, checkpoint)
    collate_fn = vg_collate_fn

  loader_kwargs = {
    'batch_size': args.batch_size,
    'num_workers': args.loader_num_workers,
    'shuffle': args.shuffle,
    'collate_fn': collate_fn,
  }
  loader = DataLoader(dset, **loader_kwargs)
  return loader


def build_model(args, checkpoint):
  kwargs = checkpoint['model_kwargs']
  model = Sg2ImModel(**checkpoint['model_kwargs'])
  model.load_state_dict(checkpoint['model_state'])
  if args.model_mode == 'eval':
    model.eval()
  elif args.model_mode == 'train':
    model.train()
  model.image_size = args.image_size
  model.cuda()
  return model


def makedir(base, name, flag=True):
  dir_name = None
  if flag:
    dir_name = os.path.join(base, name)
    if not os.path.isdir(dir_name):
      os.makedirs(dir_name)
  return dir_name

def apply_mask(image, mask, color, alpha=0.1):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image

def get_relationship(box_s, box_o, mask_s, mask_o):
    
      dic_pred = {'inside': 5, 'left of': 1, '__in_image__': 0, 'right of': 2, 'below': 4, 'above': 3, 'surrounding': 6}
      obj_centers = []
      MH, MW = mask_s.size()
      x0, y0, x1, y1 = box_s
      # mask = mask_s#(mask_s==1) #mask = (masks[i] == 1)
      mask = (mask_s==1)
      xs = torch.linspace(x0, x1, MW).view(1, MW).expand(MH, MW)
      ys = torch.linspace(y0, y1, MH).view(MH, 1).expand(MH, MW)
      if mask.sum() == 0:
        mean_x = 0.5 * (x0 + x1)
        mean_y = 0.5 * (y0 + y1)
      else:
        mean_x = xs[mask].mean()
        mean_y = ys[mask].mean()
      obj_centers.append([mean_x, mean_y])
      x0, y0, x1, y1 = box_o
      # mask = mask_o#(mask_o == 1) 
      mask = (mask_o == 1) 

      xs = torch.linspace(x0, x1, MW).view(1, MW).expand(MH, MW)
      ys = torch.linspace(y0, y1, MH).view(MH, 1).expand(MH, MW)
      if mask.sum() == 0:
        mean_x = 0.5 * (x0 + x1)
        mean_y = 0.5 * (y0 + y1)
      else:
        mean_x = xs[mask].mean()
        mean_y = ys[mask].mean()
      
      obj_centers.append([mean_x, mean_y])
      obj_centers = torch.FloatTensor(obj_centers)
    
    
    
# Check for inside / surrounding
      sx0, sy0, sx1, sy1 = box_s
      ox0, oy0, ox1, oy1 = box_o
      d = obj_centers[0] - obj_centers[1]
      theta = math.atan2(d[1], d[0])

      if sx0 < ox0 and sx1 > ox1 and sy0 < oy0 and sy1 > oy1:
        p = 'surrounding'
      elif sx0 > ox0 and sx1 < ox1 and sy0 > oy0 and sy1 < oy1:
        p = 'inside'
      elif theta >= 3 * math.pi / 4 or theta <= -3 * math.pi / 4:
        p = 'left of'
      elif -3 * math.pi / 4 <= theta < -math.pi / 4:
        p = 'above'
      elif -math.pi / 4 <= theta < math.pi / 4:
        p = 'right of'
      elif math.pi / 4 <= theta < 3 * math.pi / 4:
        p = 'below'
      
      
      p_index = dic_pred[p]
      
      return p_index
    
 
    
    
def run_model(args, checkpoint, output_dir, fn, loader=None):
  vocab = checkpoint['model_kwargs']['vocab']
  print(vocab.keys())
  print(vocab['pred_name_to_idx'])
  dic_pred = vocab['pred_name_to_idx']#{'inside': 5, 'left of': 1, '__in_image__': 0, 'right of': 2, 'below': 4, 'above': 3, 'surrounding': 6}

  model = build_model(args, checkpoint)
  if loader is None:
    loader = build_loader(args, checkpoint)

  data = {
    'vocab': vocab,
    'objs': [],
    'masks_pred': [],
    'boxes_pred': [],
    'masks_gt': [],
    'boxes_gt': [],
    'filenames': [],
  }
  which_data = args.which_data
  save_dir = makedir(output_dir, which_data)
  FH_objs_train, FH_edges_train, IDs_train = torch.load(args.FH_dir_train)#torch.load('dataFH/train_FH.npy')
  FH_objs_val, FH_edges_val, IDs_val = torch.load(args.FH_dir_val)#torch.load('dataFH/val_FH.npy')
  IDs_train = torch.tensor(IDs_train)
  IDs_val = torch.tensor(IDs_val)
  if args.which_data == 'train':
      IDs = IDs_train
      FH_objs = FH_objs_train
      FH_edges = FH_edges_train
  else:
      IDs = IDs_val
      FH_objs = FH_objs_val
      FH_edges = FH_edges_val
  
  
  
  count_edge_gt = []
  count_edge_pre = []
  img_idx = 0
  ibatch = 0
  for batch in loader:
    ibatch +=1
    masks = None
    imgs, objs, boxes, masks, triples, obj_to_img, triple_to_img, imgs_ids = [x.cuda() for x in batch]



#  get FH by images within a batch
    fh_obj, fh_edge = [],[]
    for i in range(imgs_ids.shape[0]):
          idd = ((IDs == imgs_ids[i].item()).nonzero())
          fh_obj_i = FH_objs[idd]
          fh_obj.append(fh_obj_i)
          
          fh_edge_i = FH_edges[idd]
          fh_edge.append(fh_edge_i)
          
    fh_obj = torch.cat(fh_obj)    
    fh_edge = torch.cat(fh_edge)     


    imgs_gt = imagenet_deprocess_batch(imgs)
    boxes_gt = None
    masks_gt = None
    if args.use_gt_boxes:
      boxes_gt = boxes
    if args.use_gt_masks:
      masks_gt = masks

    # Run the model with predicted masks

    model_out = model(objs, triples, fh_obj, fh_edge, obj_to_img, 
                          boxes_gt=boxes_gt, masks_gt=masks_gt)
    boxes_pred, masks_pred = model_out 


    obj_data = [objs, boxes_pred, masks_pred]
    _, obj_data = split_graph_batch(triples, obj_data, obj_to_img,
                                    triple_to_img)
    objs, boxes_pred, masks_pred = obj_data

    obj_data_gt = [boxes.data]
    if masks is not None:
      obj_data_gt.append(masks.data)
    triples, obj_data_gt = split_graph_batch(triples, obj_data_gt,
                                       obj_to_img, triple_to_img)
    boxes_gt, masks_gt = obj_data_gt[0], None
    if masks is not None:
      masks_gt = obj_data_gt[1]





    for i in range(imgs_gt.size(0)):
      # for edges  
      triples_i = triples[i]
      

      for k in range(triples_i.shape[0]):

          if(triples_i[k][1] != 0):
              
              idx_s, idx_o = triples_i[k][0], triples_i[k][2]
              
              bbxs_of_img = boxes_gt[i] 
              masks_of_img = masks_gt[i]
              box_s, box_o = bbxs_of_img[idx_s],bbxs_of_img[idx_o]
              mask_s, mask_o = masks_of_img[idx_s], masks_of_img[idx_o]
              edge_gt = get_relationship(box_s, box_o, mask_s, mask_o)
              count_edge_gt.append(edge_gt)
              # print('gt:', triples_i[k][1].item(), edge_gt)
              
              bbxs_of_img = boxes_pred[i]
              masks_of_img = masks_pred[i]
              box_s, box_o = bbxs_of_img[idx_s],bbxs_of_img[idx_o]
              mask_s, mask_o = masks_of_img[idx_s] ,masks_of_img[idx_o]
              mask_s, mask_o = torch.round(mask_s).type(torch.long), torch.round(mask_o).type(torch.long)
              edge_pre = get_relationship(box_s, box_o, mask_s, mask_o)
              count_edge_pre.append(edge_pre)
              
      
      img_idx += 1

    print('%d images' % img_idx)
            
    class2idx = {
            "left of":0,
            "right of":1,
            "above":2,
            "below":3,
            "inside":4,
            "surrounding":5
            }

    idx2class = {v: k for k, v in class2idx.items()}
      
#    break
    

  print('gt',len(count_edge_gt))
  print('pre',len(count_edge_pre))
  cm = confusion_matrix( count_edge_pre,count_edge_gt)  # y, x
  cm = cm/cm.sum(axis = 0)
  confusion_matrix_df = pd.DataFrame(cm).rename(columns=idx2class, index=idx2class)
  label = {'a': '5%', 'b':'10%','c': '20%', 'd':'50%','e': '100%'}
  ax = sns.heatmap(confusion_matrix_df, annot=True, cmap='Blues_r', vmin=0, vmax=1)
  title = 'M1_bm_FH_'+ args.which_data + '_' + label[fn] 
  ax.set(title=title,
             ylabel='Predicted label',
             xlabel='True label')
  fig = ax.get_figure()
  filename = 'CM1_bm_FH_' + fn + '_' + args.which_data + '.png'
  CM_path = os.path.join(output_dir,args.which_data, filename)
  fig.savefig(CM_path)
  fig.clf()
  print('over')
  

def main(args):
  got_checkpoint = args.checkpoint is not None
  got_checkpoint_list = args.checkpoint_list is not None
  if got_checkpoint == got_checkpoint_list:
    raise ValueError('Must specify exactly one of --checkpoint and --checkpoint_list')

  if got_checkpoint:
    checkpoint = torch.load(args.checkpoint)
    print('Loading model from ', args.checkpoint)
    run_model(args, checkpoint, args.output_dir, args.checkpoint)
  elif got_checkpoint_list:
    # For efficiency, use the same loader for all checkpoints
    label = ['a', 'b', 'c', 'd','e','f']
    loader = None
    with open(args.checkpoint_list, 'r') as f:
      checkpoint_list = [line.strip() for line in f]
    for i, path in enumerate(checkpoint_list):
      if os.path.isfile(path):
        print('Loading model from ', path)
        checkpoint = torch.load(path)
        if loader is None:
          loader = build_loader(args, checkpoint)
        output_dir = os.path.join(args.output_dir, 'result_CM1_bm_FH')
        run_model(args, checkpoint, output_dir, label[i], loader)
      elif os.path.isdir(path):
        # Look for snapshots in this dir
        i = 0
        for fn in sorted(os.listdir(path)):
          if 'snapshot' not in fn:
            continue
          checkpoint_path = os.path.join(path, fn)
          print('Loading model from ', checkpoint_path)
          checkpoint = torch.load(checkpoint_path)
          if loader is None:
            loader = build_loader(args, checkpoint)

          # Snapshots have names like "snapshot_00100K.pt'; we want to
          # extract the "00100K" part
          snapshot_name = os.path.splitext(fn)[0].split('_')[1]
          output_dir = 'results'#%03d_%s' % (i, snapshot_name)
          output_dir = os.path.join(args.output_dir, output_dir)

          run_model(args, checkpoint, output_dir, label[i], loader)
          i = i+1


if __name__ == '__main__':
  args = parser.parse_args()
  main(args)


