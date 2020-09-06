# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Xinlei Chen
# --------------------------------------------------------

"""Compute minibatch blobs for training a Fast R-CNN network."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import numpy.random as npr
from scipy.misc import imread, imresize
from model.utils.config import cfg
from model.utils.blob import prep_im_for_blob, im_list_to_blob
import pdb

def get_minibatch(roidb, num_classes,seg_return=False):
  """Given a roidb, construct a minibatch sampled from it."""
  num_images = len(roidb)
  # Sample random scales to use for each image in this batch
  random_scale_inds = npr.randint(0, high=len(cfg.TRAIN.SCALES),
                  size=num_images)
  assert(cfg.TRAIN.BATCH_SIZE % num_images == 0), \
    'num_images ({}) must divide BATCH_SIZE ({})'. \
    format(num_images, cfg.TRAIN.BATCH_SIZE)

  # Get the input image blob, formatted for caffe
  im_blob, im_scales = _get_image_blob(roidb, random_scale_inds)

  blobs = {'data': im_blob}
  # blobs['ps_data'] = ps_im_blob

  assert len(im_scales) == 1, "Single batch only"
  assert len(roidb) == 1, "Single batch only"
  
  # gt boxes: (x1, y1, x2, y2, cls)
  if cfg.TRAIN.USE_ALL_GT:
    # Include all ground truth boxes
    gt_inds = np.where(roidb[0]['gt_classes'] != 0)[0]
  else:
    # For the COCO ground truth boxes, exclude the ones that are ''iscrowd'' 
    gt_inds = np.where((roidb[0]['gt_classes'] != 0) & np.all(roidb[0]['gt_overlaps'].toarray() > -1.0, axis=1))[0]
  gt_boxes = np.empty((len(gt_inds), 5), dtype=np.float32)
  gt_boxes[:, 0:4] = roidb[0]['boxes'][gt_inds, :] * im_scales[0]
  gt_boxes[:, 4] = roidb[0]['gt_classes'][gt_inds]
  blobs['gt_boxes'] = gt_boxes

  blobs['im_info'] = np.array(
    [[im_blob.shape[1], im_blob.shape[2], im_scales[0]]],
    dtype=np.float32)

  if seg_return:
    blobs['seg_map'] = roidb[0]['seg_map']

  blobs['img_id'] = roidb[0]['img_id']
  blobs['path'] = roidb[0]['image']

  # pdb.set_trace()
  # print(im_blob)

  return blobs

def _get_image_blob(roidb, scale_inds):
  """Builds an input blob from the images in the roidb at the specified
  scales.
  """
  num_images = len(roidb)
  processed_ims = []
  processed_ps_ims = []

  im_scales = []
  for i in range(num_images):
    # im = cv2.imread(roidb[i]['image'])
    im = imread(roidb[i]['image'])
    # ps_im = imread(roidb[i]['ps_image'])

    # if im.shape != ps_im.shape:
    #   im=imresize(im, size=ps_im.shape, interp='bicubic')


    if len(im.shape) == 2:
      im = im[:,:,np.newaxis]
      im = np.concatenate((im,im,im), axis=2)

    # if len(ps_im.shape) == 2:
    #   ps_im = ps_im[:,:,np.newaxis]
    #   ps_im = np.concatenate((ps_im,ps_im,ps_im), axis=2)

    # flip the channel, since the original one using cv2
    # rgb -> bgr
    im = im[:,:,::-1]
    
    # pdb.set_trace()

    if roidb[i]['flipped']:
      im = im[:, ::-1, :]
      # ps_im = ps_im[:, ::-1, :]

    target_size = cfg.TRAIN.SCALES[scale_inds[i]]
    im, im_scale = prep_im_for_blob(im, cfg.PIXEL_MEANS, target_size,
                    cfg.TRAIN.MAX_SIZE)

    im_scales.append(im_scale)
    processed_ims.append(im)

    # processed_ps_ims.append(ps_im)

  # Create a blob to hold the input images
  blob = im_list_to_blob(processed_ims)
  # ps_blob = im_list_to_blob(processed_ps_ims)

  # return blob, ps_blob, im_scales
  
  return blob, im_scales