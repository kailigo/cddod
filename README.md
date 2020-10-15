# An example using the proposed benchmark suite


## Introduction
The code is heavily developed based on this [repo](https://github.com/VisionLearningGroup/DA_Detection), which is based on 
the [faster-rcnn repository](https://github.com/jwyang/faster-rcnn.pytorch). Refer these two repos for environment setup. 

## Prepare the data
* Download the "Chs" and "PubMed" datasets [here](https://drive.google.com/file/d/1m4ns2gbl3d4fcms5Ta6IS80g964sn_b2/view?usp=sharing). The datasets have been organized in the "Pascal_VOC" format.

* write the path in __C.VGG_PATH and __C.RESNET_PATH at lib/model/utils/config.py.


## Train
* Adaptation from "PubMed" to "Chs"
```
CUDA_VISIBLE_DEVICES=$GPU_ID python trainval_net_global_local.py --cuda --net res101 --dataset doc_med --dataset_t doc_chs_median --save_dir ./models/ckpts/da_med2chsmed_ps_lr3ss8 --stdout_file da_med2chsmed_ps_lr3ss8 --lr 1e-3 --lr_decay_step 8
```

* Adaptation from "PubMed" to "Chs"
```
CUDA_VISIBLE_DEVICES=$GPU_ID python trainval_net_global_local.py --cuda --net res101 --dataset doc_chs_median --dataset_t doc_med --save_dir ./models/ckpts/da_chsmed2med_ps_lr3ss8 --stdout_file da_med2chsmed_ps_lr3ss8 --lr 1e-3 --lr_decay_step 8
```

## Citation
Please cite the following reference if you find this repo helps your research.

```
@inproceedings{li2020cross,
  title={Cross-Domain Document Object Detection: Benchmark Suite and Method},
  author={Li, Kai and Wigington, Curtis and Tensmeyer, Chris and Zhao, Handong and Barmpalios, Nikolaos and Morariu, Vlad I and Manjunatha, Varun and Sun, Tong and Fu, Yun},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={12915--12924},
  year={2020}
}
```


