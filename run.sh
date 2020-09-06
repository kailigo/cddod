
CUDA_VISIBLE_DEVICES=3 python trainval_net_global_local.py --cuda --net res101 --dataset doc_med --dataset_t doc_chs_median --save_dir ./models/ckpts/da_med2chsmed_ps_lr3ss8 --stdout_file da_med2chsmed_ps_lr3ss8 --lr 1e-3 --lr_decay_step 8

CUDA_VISIBLE_DEVICES=2 python trainval_net_global_local.py --cuda --net res101 --dataset doc_chs_median --dataset_t doc_med --save_dir ./models/ckpts/da_chsmed2med_ps_lr3ss8 --stdout_file da_med2chsmed_ps_lr3ss8 --lr 1e-3 --lr_decay_step 8
