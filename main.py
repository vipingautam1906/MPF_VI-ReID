import os
import glob
import pprint
import logging

import numpy as np
import scipy.io as sio
from ignite.engine import Events

import torch
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler

from data import get_train_loader, get_test_loader
from models.model import PMRNet
from engine import get_trainer
from engine.engine import create_eval_engine
from utils.eval_data import eval_sysu, eval_regdb, eval_llcm
from utils.helper import log_model_parameters
from utils.aqe import aqe
import torch.nn.functional as F


# =========================================================
# TRAIN
# =========================================================
def train(cfg, args):
    logger = logging.getLogger('PMRNet')

    tb_dir = os.path.join(cfg.log_dir, 'tensorboard')
    os.makedirs(tb_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=tb_dir)

    # ---------------------- Data ---------------------------
    train_loader = get_train_loader(
        dataset=cfg.dataset,
        root=cfg.data_root,
        sample_method=cfg.sample_method,
        batch_size=cfg.batch_size,
        p_size=cfg.p_size,
        k_size=cfg.k_size,
        image_size=cfg.image_size,
        random_flip=cfg.random_flip,
        random_crop=cfg.random_crop,
        random_erase=cfg.random_erase,
        color_jitter=cfg.color_jitter,
        padding=cfg.padding,
        tms=cfg.tms,
        num_workers=8
    )

    gallery_loader, query_loader = None, None
    if cfg.eval_interval > 0:
        gallery_loader, query_loader = get_test_loader(
            dataset=cfg.dataset,
            root=cfg.data_root,
            batch_size=cfg.batch_size,
            image_size=cfg.image_size,
            num_workers=8
        )

    model = PMRNet(
            num_classes=cfg.num_id,
            dataset=cfg.dataset,
            num_parts=cfg.num_parts,
            parts=cfg.parts,
            PMR=cfg.PMR,
            prototypes=cfg.prototypes,
            MAR=cfg.MAR,
            alignment=cfg.alignment,
            p = cfg.train_p,
        ).cuda()
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=cfg.lr_step, gamma=0.1
    )

    scaler = GradScaler(enabled=cfg.fp16)

    # ---------------------- Resume -------------------------
    if cfg.resume:
        checkpoint = torch.load(cfg.resume)
        model_dict = model.state_dict()
        for k in list(checkpoint.keys()):
            if k in model_dict and checkpoint[k].shape != model_dict[k].shape:
                logger.info(f'Remove mismatched key: {k}')
                checkpoint.pop(k)
        model.load_state_dict(checkpoint, strict=False)

    # ---------------------- Engine -------------------------
    checkpoint_dir = os.path.join('ckptlog/', cfg.dataset, cfg.prefix)

    engine = get_trainer(
        dataset=cfg.dataset,
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        logger=logger,
        writer=writer,
        non_blocking=True,
        log_period=cfg.log_period,
        save_dir=checkpoint_dir,
        prefix=cfg.prefix,
        eval_interval=cfg.eval_interval,
        start_eval=cfg.start_eval,
        gallery_loader=gallery_loader,
        query_loader=query_loader
    )

    @engine.on(Events.EPOCH_STARTED)
    def setter_epoch(engine):
        epoch = engine.state.epoch
        model.current_epoch = epoch
    engine.run(train_loader, max_epochs=cfg.num_epoch)
    writer.close()

@torch.no_grad()
def test(cfg, args):
    logger = logging.getLogger('PMRNet')
    logger.info('\n## Testing...')
    gallery_loader, query_loader = get_test_loader(
        dataset=cfg.dataset,
        root=cfg.data_root,
        batch_size=cfg.batch_size,
        image_size=cfg.image_size,
        num_workers=8
    )

    model = PMRNet(
            num_classes=cfg.num_id,
            dataset=cfg.dataset,
            num_parts=cfg.num_parts,
            parts=cfg.parts,
            PMR=cfg.PMR,
            prototypes=cfg.prototypes,
            MAR=cfg.MAR,
            alignment=cfg.alignment,
            p = cfg.test_p,
        ).cuda()
    

    resume_path = cfg.resume if cfg.resume else glob.glob(f'{cfg.log_dir}/*best*')[0]
    checkpoint = torch.load(resume_path)
    model.load_state_dict(checkpoint, strict=False)
    model.eval()
    log_model_parameters(model)
    evaluator = create_eval_engine(model, non_blocking=True)

    with autocast(enabled=cfg.fp16):
        evaluator.run(query_loader)
        q_feats = torch.cat(evaluator.state.feat_list)
        q_ids = torch.cat(evaluator.state.id_list).numpy()
        q_cams = torch.cat(evaluator.state.cam_list).numpy()
        q_img_paths = np.concatenate(evaluator.state.img_path_list)

        evaluator.run(gallery_loader)
        g_feats = torch.cat(evaluator.state.feat_list)
        g_ids = torch.cat(evaluator.state.id_list).numpy()
        g_cams = torch.cat(evaluator.state.cam_list).numpy()
        g_img_paths = np.concatenate(evaluator.state.img_path_list)
        q_feats = aqe(q_feats, g_feats, k=5, alpha=3.0) if cfg.aqe else q_feats


    if cfg.dataset == 'sysu':
        perm = sio.loadmat(os.path.join(cfg.data_root, 'exp', 'rand_perm_cam.mat'))['rand_perm_cam']
        
        eval_sysu(q_feats, q_ids, q_cams, q_img_paths,
                  g_feats, g_ids, g_cams, g_img_paths, perm, mode='all', num_shots=1, rerank=cfg.rerank)
        # eval_sysu(q_feats, q_ids, q_cams, q_img_paths,
        #           g_feats, g_ids, g_cams, g_img_paths, perm, mode='all', num_shots=10, rerank=cfg.rerank)

        eval_sysu(q_feats, q_ids, q_cams, q_img_paths, 
                  g_feats, g_ids, g_cams, g_img_paths, perm, mode='indoor', num_shots=1, rerank=cfg.rerank)
        # eval_sysu(q_feats, q_ids, q_cams, q_img_paths,
        #               g_feats, g_ids, g_cams, g_img_paths, perm, mode='indoor', num_shots=10, rerank=cfg.rerank)

    elif cfg.dataset == 'regdb':
        eval_regdb(g_feats, g_ids, g_cams, g_img_paths,
                   q_feats, q_ids, q_cams, q_img_paths, mode='v2i', rerank=cfg.rerank)
        eval_regdb(q_feats, q_ids, q_cams, q_img_paths,
                   g_feats, g_ids, g_cams, g_img_paths, mode='i2v', rerank=cfg.rerank)
    elif cfg.dataset == 'llcm':
        logger.info('Test Mode - visible to infrared')
        eval_llcm(
            g_feats, g_ids, g_cams, g_img_paths,
            q_feats, q_ids, q_cams, q_img_paths,
            mode='v2i',
            rerank=cfg.rerank
        )
        logger.info('Test Mode - infrared to visible')
        eval_llcm(
            q_feats, q_ids, q_cams, q_img_paths,
            g_feats, g_ids, g_cams, g_img_paths,
            mode='i2v',
            rerank=cfg.rerank
        )
    else:
        raise NotImplementedError(cfg.dataset)


if __name__ == '__main__':
    import argparse
    from configs.default import strategy_cfg, dataset_cfg
    from utils.tools import set_seed, time_str

    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='configs/SYSU.yml')
    parser.add_argument('--seed', type=int, default=8)
    parser.add_argument('--gpu', type=str, default='0')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    set_seed(args.seed)

    cfg = strategy_cfg
    cfg.merge_from_file(args.cfg)
    for k, v in dataset_cfg.get(cfg.dataset).items():
        cfg[k] = v

    cfg.prefix += f'_Time-{time_str()}'
    cfg.log_dir = os.path.join('ckptlog/', cfg.dataset, cfg.prefix)
    cfg.freeze()

    os.makedirs(cfg.log_dir, exist_ok=True)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logging.getLogger("ignite").setLevel(logging.WARNING)

    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('[%(asctime)s] %(message)s'))
    logger.addHandler(handler)

    if not cfg.test_only:
        train(cfg, args)
    test(cfg, args)
