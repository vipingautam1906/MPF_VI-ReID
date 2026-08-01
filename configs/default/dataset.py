from yacs.config import CfgNode

dataset_cfg = CfgNode()

dataset_cfg.sysu = CfgNode()
dataset_cfg.sysu.num_id = 395
dataset_cfg.sysu.num_cam = 6
dataset_cfg.sysu.data_root = "/home/vipin/PhD/datasets/cross_modality/SYSU-MM01/SYSU-MM01"

dataset_cfg.regdb = CfgNode()
dataset_cfg.regdb.num_id = 206
dataset_cfg.regdb.num_cam = 2
dataset_cfg.regdb.data_root = "/home/vipin/PhD/datasets/cross_modality/RegDB/RegDB"


dataset_cfg.llcm = CfgNode()
dataset_cfg.llcm.num_id = 713
dataset_cfg.llcm.num_cam = 2
dataset_cfg.llcm.data_root = "/home/vipin/PhD/datasets/cross_modality/LLCM"