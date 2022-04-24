from main import main

import os
from glob import glob
from inference import inference_main


class CFG:
    n_split = 5
    fold = 0
    seed = 2022
    epoch = 21
    weight_decay = 0.
    s = 30.0
    model_name = 'tf_efficientnet_b5_ns'
    train_data = '../data/train.csv'
    train_image_root = '../data/train_detec_512_v5/'
    image_size = 16, 16
    batch_size = 512
    iters_to_accumulate = 1
    learning_rate = 5e-6 * batch_size * iters_to_accumulate
    emb_size = 2048
    m = 0.55
    
    
if __name__ == '__main__':
    config = CFG()
    config.test = False
    config.full = False
    config.wandb_log = True
    config.gamma = 1
    config.emb_size = 2048 # 512
    config.epoch = 21
    config.s = 30
    config.m = 0.55

    config.train_data = '/USER/data3/train.csv'
    config.train_image_root = '/USER/beluga/'
    config.project = 'beluga'

    config.batch_size = 32
    config.iters_to_accumulate = 2
    config.learning_rate = 6.25e-6 * config.batch_size * config.iters_to_accumulate
    config.model_name = 'convnext_base' # 'tf_efficientnet_b5_ap', 'tf_efficientnetv2_m'
    config.image_size = 224, 512 # 128*512, 512*2048
    config.exp_name = '384_v2_convnext_gauss5'

    config.fold = 0
    main(config)