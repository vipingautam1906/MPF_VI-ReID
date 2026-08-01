from yacs.config import CfgNode

strategy_cfg = CfgNode()

strategy_cfg.prefix = "SYSU"

# Settings for Data
strategy_cfg.dataset = "sysu"
strategy_cfg.image_size = (384, 128)
strategy_cfg.sample_method = "norm_triplet"
strategy_cfg.p_size = 8
strategy_cfg.k_size = 16
strategy_cfg.batch_size = 128

# Settings for Augmentation
strategy_cfg.random_flip = True
strategy_cfg.random_crop = True
strategy_cfg.random_erase = True
strategy_cfg.color_jitter = True
strategy_cfg.padding = 10
strategy_cfg.tms = True

# Settings for Model
strategy_cfg.num_parts = 6
strategy_cfg.parts = True
strategy_cfg.PMR = True
strategy_cfg.prototypes = True
strategy_cfg.MAR = True
strategy_cfg.alignment = True

strategy_cfg.drop_last_stride = False

# Setting for Loss
strategy_cfg.loss_id = True

# Settings for Training
strategy_cfg.lr = 0.00035
strategy_cfg.wd = 0.0005
strategy_cfg.num_epoch = 140
strategy_cfg.lr_step = [80, 120]
strategy_cfg.fp16 = True
strategy_cfg.train_p = 3.0
strategy_cfg.test_p = 3.5

# Settings for Logging
strategy_cfg.start_eval = 0
strategy_cfg.log_period = 10
strategy_cfg.eval_interval = 1

# Settings for Testing
strategy_cfg.resume = ''
strategy_cfg.test_only = False
strategy_cfg.rerank = False
strategy_cfg.aqe = False