# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Factory method for easily getting imdbs by name."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__sets = {}


from datasets.doc_med import doc_med

from datasets.doc_chs_median import doc_chs_median

import numpy as np





for split in ['train', 'trainval', 'val','test']:
  name = 'cityscape_{}'.format(split)
  __sets[name] = (lambda split=split : cityscape(split))

for split in ['train', 'trainval','val','test']:
  name = 'cityscape_car_{}'.format(split)
  __sets[name] = (lambda split=split : cityscape_car(split))

for split in ['train', 'trainval','test']:
  name = 'foggy_cityscape_{}'.format(split)
  __sets[name] = (lambda split=split : foggy_cityscape(split))

for split in ['train','val']:
  name = 'sim10k_{}'.format(split)
  __sets[name] = (lambda split=split : sim10k(split))

for split in ['train', 'val']:
  name = 'sim10k_cycle_{}'.format(split)
  __sets[name] = (lambda split=split: sim10k_cycle(split))

for year in ['2007', '2012']:
  for split in ['train', 'val', 'trainval', 'test']:
    name = 'voc_{}_{}'.format(year, split)
    __sets[name] = (lambda split=split, year=year: pascal_voc(split, year))
for year in ['2007', '2012']:
  for split in ['train', 'val', 'trainval', 'test']:
    name = 'voc_water_{}_{}'.format(year, split)
    __sets[name] = (lambda split=split, year=year: pascal_voc_water(split, year))
for year in ['2007', '2012']:
  for split in ['train', 'val', 'trainval', 'test']:
      name = 'voc_cycleclipart_{}_{}'.format(year, split)
      __sets[name] = (lambda split=split, year=year: pascal_voc_cycleclipart(split, year))
for year in ['2007', '2012']:
  for split in ['train', 'val', 'trainval', 'test']:
      name = 'voc_cyclewater_{}_{}'.format(year, split)
      __sets[name] = (lambda split=split, year=year: pascal_voc_cyclewater(split, year))
for year in ['2007']:
  for split in ['trainval', 'test']:
    name = 'clipart_{}'.format(split)
    __sets[name] = (lambda split=split : clipart(split,year))
for year in ['2007']:
  for split in ['train', 'test']:
    name = 'water_{}'.format(split)
    __sets[name] = (lambda split=split : water(split,year))


for split in ['train', 'trainval', 'val', 'test']:
  name = 'pure_{}'.format(split)
  __sets[name] = (lambda split=split : pure(split))

for split in ['train', 'trainval', 'val','test']:
  name = 'hive_{}'.format(split)
  __sets[name] = (lambda split=split : hive(split))


for split in ['train', 'trainval', 'val','test']:
  name = 'doc_financial_report_{}'.format(split)
  __sets[name] = (lambda split=split : doc_financial_report(split))


for split in ['train', 'trainval', 'val','test']:
  name = 'doc_legal_{}'.format(split)
  __sets[name] = (lambda split=split : doc_legal(split))


for split in ['train', 'trainval', 'val','test']:
  name = 'doc_chs_median_{}'.format(split)
  __sets[name] = (lambda split=split : doc_chs_median(split))


for split in ['train', 'trainval', 'val','test']:
  name = 'doc_chs_{}'.format(split)
  __sets[name] = (lambda split=split : doc_chs(split))


for split in ['train', 'trainval', 'val','test']:
  name = 'doc_med_{}'.format(split)
  __sets[name] = (lambda split=split : doc_med(split))



def get_imdb(name):
  """Get an imdb (image database) by name."""
  if name not in __sets:
    raise KeyError('Unknown dataset: {}'.format(name))
  return __sets[name]()


def list_imdbs():
  """List all registered imdbs."""
  return list(__sets.keys())
