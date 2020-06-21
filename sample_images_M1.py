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
import os
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

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
# parser.add_argument('--model', default='bm', choices=['bm', 'bm_FH', 'bm_FH64', 'bm_FHrec', 'bm_FHrec64'])

# Shared dataset options
parser.add_argument('--dataset', default='coco', choices=['coco', 'vg'])
parser.add_argument('--which_data', default='val', choices=['train', 'val'])

parser.add_argument('--image_size', default=(64, 64), type=int_tuple)
parser.add_argument('--batch_size', default=1, type=int)
parser.add_argument('--shuffle', default=False, type=bool_flag)
parser.add_argument('--loader_num_workers', default=4, type=int)
parser.add_argument('--num_samples', default=10000, type=int)
parser.add_argument('--save_gt_imgs', default=False, type=bool_flag)
parser.add_argument('--save_graphs', default=False, type=bool_flag)
parser.add_argument('--use_gt_boxes', default=False, type=bool_flag)
parser.add_argument('--use_gt_masks', default=False, type=bool_flag)
parser.add_argument('--save_layout', default=True, type=bool_flag)
parser.add_argument('--output_dir', default='output_samples')



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

#colors = [(250,128,114),(32,178,170),(240,128,128),(248,248,255),(175,238,238), (176,224,230),(15,38,238), (76,124,100) ,(38,105,238), (224,76,100)]
colors = [(255,0,0),(255,128,0),(255,255,0),(128,255,0),(0,255,255), (0,128,255),(0,0,255),(128,0,255),(255,0,255),(255,0,128)]

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

def run_model(args, checkpoint, output_dir, fn, loader=None):
  vocab = checkpoint['model_kwargs']['vocab']
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
#  save_dir = makedir(save_dir,fn)
  
  img_idx = 0
  ibatch = 0
  for batch in loader:
    ibatch +=1
    masks = None
    if len(batch) == 6:
      imgs, objs, boxes, triples, obj_to_img, triple_to_img = [x.cuda() for x in batch]
    elif len(batch) == 7:
      imgs, objs, boxes, masks, triples, obj_to_img, triple_to_img = [x.cuda() for x in batch]
    # imgs, objs, boxes, masks, triples, obj_to_img, triple_to_img, imgs_ids = [x.cuda() for x in batch]
    
#    imgs_print = imagenet_deprocess_batch(imgs)
#    grid = torchvision.utils.make_grid(imgs_print)
#    writer.add_image('img/real', grid, ibatch-1)
    imgs_gt = imagenet_deprocess_batch(imgs)
    boxes_gt = None
    masks_gt = None
    if args.use_gt_boxes:
      boxes_gt = boxes
    if args.use_gt_masks:
      masks_gt = masks

    # Run the model with predicted masks
    model_out = model(objs, triples, obj_to_img,
                          boxes_gt=boxes_gt, masks_gt=masks_gt)
    # boxes_pred, masks_pred = model_out 
    imgs_pred, boxes_pred, masks_pred, _ = model_out


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

    imgs_bbx = torch.zeros(imgs_gt.size(),dtype=torch.uint8)
    imgs_bbx_pre = torch.zeros(imgs_gt.size(),dtype=torch.uint8)
    white_bbx_gt, white_bbx_gtb = torch.zeros(imgs_gt.size(),dtype=torch.uint8), torch.zeros(imgs_gt.size(),dtype=torch.uint8)
    white_bbx_pre, white_bbx_preb = torch.zeros(imgs_gt.size(),dtype=torch.uint8), torch.zeros(imgs_gt.size(),dtype=torch.uint8)

    
    for i in range(imgs_gt.size(0)):

      black_gt = np.zeros([args.image_size[0],args.image_size[1],3]) 
      black_gtb = np.zeros([args.image_size[0],args.image_size[1],3]) 
      img = np.copy(imgs_gt[i].numpy().transpose(1, 2, 0))
      layer = np.zeros(list(args.image_size))    
      masks_of_img = masks_gt[i]
      bbxs_of_img = boxes_gt[i]
      num_of_objs = bbxs_of_img.size(0)
      for j in range(num_of_objs-1):   

#          color = tuple(np.random.randint(256, size=3))
          color = colors[j%len(colors)]
          mask = masks_of_img[j].cpu().clone().numpy()
          mask = np.round(mask)
          bbx = (bbxs_of_img[j].cpu().numpy() * args.image_size[0]).astype(int)
          bbx = np.clip(bbx, 0,args.image_size[0]-1)
          wbbx = bbx[2] - bbx[0]
          hbbx = bbx[3] - bbx[1]
          if not wbbx >0:
              wbbx = 1
              print('gt',wbbx,hbbx)
          if not hbbx >0:
              hbbx = 1
              print('gt',wbbx,hbbx)          
          maskPIL = Image.fromarray(mask.astype(np.uint8))
          maskPIL = maskPIL.resize((wbbx,hbbx), resample=Image.BILINEAR)
            
          layer[ bbx[1]:bbx[3],bbx[0]:bbx[2]] = np.array(maskPIL)
          img = apply_mask(img, layer, color)
          masked_imgPIL = Image.fromarray(img.astype(np.uint8))
          draw = ImageDraw.Draw(masked_imgPIL)          
          draw.rectangle(bbx.tolist(), width=1, outline = color)
          img = np.array(masked_imgPIL)
          
          black_gt = apply_mask(black_gt, layer, color)
          masked_blackPIL = Image.fromarray(black_gt.astype(np.uint8))
          draw2 = ImageDraw.Draw(masked_blackPIL)
          draw2.rectangle(bbx.tolist(), width=1, outline = color) 
          black_gt = np.array(masked_blackPIL)
          
          blackPIL = Image.fromarray(black_gtb.astype(np.uint8))
          draw2b = ImageDraw.Draw(blackPIL)
          draw2b.rectangle(bbx.tolist(), width=1, outline = color)
          black_gtb = np.array(blackPIL)
          
      imgs_bbx[i] = torchvision.transforms.ToTensor()(masked_imgPIL)*255  
      white_bbx_gt[i] = torchvision.transforms.ToTensor()(masked_blackPIL)*255 
      white_bbx_gtb[i] = torchvision.transforms.ToTensor()(blackPIL)*255 
      
      black_gt = np.zeros([args.image_size[0],args.image_size[1],3]) 
      black_gtb = np.zeros([args.image_size[0],args.image_size[1],3]) 
      img = np.copy(imgs_gt[i].numpy().transpose(1, 2, 0))
      layer = np.zeros(list(args.image_size))    
      bbxs_of_img = boxes_pred[i]
      masks_of_img = masks_pred[i]
      num_of_objs = bbxs_of_img.size(0)
      for j in range(num_of_objs-1): 

          color = colors[j%len(colors)]
          
          mask = masks_of_img[j].cpu().clone().numpy()
          mask = np.round(mask)
          bbx = (bbxs_of_img[j].cpu().numpy() * args.image_size[0]).astype(int)
          bbx = np.clip(bbx, 0,args.image_size[0]-1)
          wbbx = bbx[2] - bbx[0]
          hbbx = bbx[3] - bbx[1]
          if not wbbx >0:
              wbbx = 1
              print('pred',wbbx,hbbx)
          if not hbbx >0:
              hbbx = 1
              print('pred',wbbx,hbbx)
          maskPIL = Image.fromarray(mask.astype(np.uint8))
          maskPIL = maskPIL.resize((wbbx,hbbx), resample=Image.BILINEAR)
#          print('wwbx,hbbx:',wbbx, hbbx, maskPIL2.size, bbx)
          layer[ bbx[1]:bbx[3],bbx[0]:bbx[2]] = np.array(maskPIL)
          img = apply_mask(img, layer, color)
          masked_imgPIL = Image.fromarray(img.astype(np.uint8))
          draw = ImageDraw.Draw(masked_imgPIL)          
          draw.rectangle(bbx.tolist(), width=1, outline = color)
          img = np.array(masked_imgPIL)
          
          black_gt = apply_mask(black_gt, layer, color)
          masked_blackPIL = Image.fromarray(black_gt.astype(np.uint8))
          draw2 = ImageDraw.Draw(masked_blackPIL)
          draw2.rectangle(bbx.tolist(), width=1, outline = color) 
          black_gt = np.array(masked_blackPIL)
          
          blackPIL = Image.fromarray(black_gtb.astype(np.uint8))
          draw2b = ImageDraw.Draw(blackPIL)
          draw2b.rectangle(bbx.tolist(), width=1, outline = color)
          black_gtb = np.array(blackPIL)
          
      imgs_bbx_pre[i] = torchvision.transforms.ToTensor()(masked_imgPIL)*255  
      white_bbx_pre[i] = torchvision.transforms.ToTensor()(masked_blackPIL)*255  
      white_bbx_preb[i] = torchvision.transforms.ToTensor()(blackPIL)*255 

      img_idx += 1
      
    imgs_orig = imagenet_deprocess_batch(imgs)
    grid1 = torchvision.utils.make_grid(imgs_orig)
    toSave = grid1
     # GT
    # imgs_grid_GT = imgs_bbx.byte()
    # grid2 = torchvision.utils.make_grid(imgs_grid_GT)

    # toSave = torch.cat((grid1,grid2),1)
    
    
    white_grid_GT = white_bbx_gt.byte()
    grid3 = torchvision.utils.make_grid(white_grid_GT)

    toSave = torch.cat((toSave,grid3),1)
    
    white_grid_GTb = white_bbx_gtb.byte()
    grid3b = torchvision.utils.make_grid(white_grid_GTb)

    toSave = torch.cat((toSave,grid3b),1)
    # PRE
    imgs_pred = imagenet_deprocess_batch(imgs_pred)
    gridx = torchvision.utils.make_grid(imgs_pred)
    toSave = torch.cat((toSave, gridx),1)
    
    # imgs_grid_pre = imgs_bbx_pre.byte()
    # grid4 = torchvision.utils.make_grid(imgs_grid_pre)

    # toSave = torch.cat((toSave, grid4),1)
    
    white_grid_pre = white_bbx_pre.byte()
    grid5 = torchvision.utils.make_grid(white_grid_pre)
    
    toSave = torch.cat((toSave, grid5),1)
    
    white_grid_preb = white_bbx_preb.byte()
    grid5b = torchvision.utils.make_grid(white_grid_preb)
    
    toSave = torch.cat((toSave, grid5b),1)
    
    toSavePIL = torchvision.transforms.ToPILImage()(toSave)

      
    save_dir = 'output'
    fn = 'M1re'
    grids_path = os.path.join(save_dir, '%d'%img_idx + fn + '.png')    
    # grids_path = os.path.join(save_dir, '%d'%img_id + fn + '.png')
    toSavePIL.save(grids_path)
    print('Saved %d images' % img_idx)
    
  

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
        output_dir = os.path.join(args.output_dir, 'samples_'+ args.model) 
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

