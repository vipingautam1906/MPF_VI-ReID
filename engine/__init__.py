import os

import numpy as np
import scipy.io as sio
import torch

from glob import glob
from ignite.engine import Events
from ignite.handlers import ModelCheckpoint
from ignite.handlers import Timer

from engine.engine import create_eval_engine
from engine.engine import create_train_engine
from engine.metric import AutoKVMetric
from utils.eval_data import eval_sysu, eval_regdb, eval_llcm
from configs.default.dataset import dataset_cfg


def get_trainer(dataset, model, optimizer, lr_scheduler=None, logger=None, writer=None, non_blocking=False,
                log_period=10, save_dir="checkpoints", prefix="model", eval_interval=None, start_eval=None,
                gallery_loader=None, query_loader=None, rerank=False):
    # Trainer
    trainer = create_train_engine(model, optimizer, non_blocking)
    setattr(trainer, "rerank", rerank)

    # Checkpoint Handler
    # handler = ModelCheckpoint(save_dir, prefix, save_interval=eval_interval, n_saved=3, create_dir=True,
    #                           save_as_state_dict=True, require_empty=False)
    
    # trainer.add_event_handler(Events.EPOCH_COMPLETED, handler, {"model": model})

    handler = ModelCheckpoint(
        dirname=save_dir,
        filename_prefix=prefix,
        n_saved=3,
        create_dir=True,
        require_empty=False
    )

    trainer.add_event_handler(
        Events.EPOCH_COMPLETED(every=eval_interval),
        handler,
        {"model": model}
    )


    timer = Timer(average=True)
    kv_metric = AutoKVMetric()

    # Evaluator
    evaluator = None
    if not type(eval_interval) == int:
        raise TypeError("The parameter 'validate_interval' must be type INT.")
    if not type(start_eval) == int:
        raise TypeError("The parameter 'start_eval' must be type INT.")
    if eval_interval > 0 and gallery_loader is not None and query_loader is not None:
        evaluator = create_eval_engine(model, non_blocking)

    def run_init_eval(engine):
        logger.info('\n## Checking model performance with initial parameters...')

        # Extract Query Feature
        evaluator.run(query_loader)
        q_feats = torch.cat(evaluator.state.feat_list, dim=0)
        q_ids = torch.cat(evaluator.state.id_list, dim=0).numpy()
        q_cams = torch.cat(evaluator.state.cam_list, dim=0).numpy()
        q_img_paths = np.concatenate(evaluator.state.img_path_list, axis=0)

        # Extract Gallery Feature
        evaluator.run(gallery_loader)
        g_feats = torch.cat(evaluator.state.feat_list, dim=0)
        g_ids = torch.cat(evaluator.state.id_list, dim=0).numpy()
        g_cams = torch.cat(evaluator.state.cam_list, dim=0).numpy()
        g_img_paths = np.concatenate(evaluator.state.img_path_list, axis=0)

        if dataset == 'sysu':
            perm = sio.loadmat(os.path.join(dataset_cfg.sysu.data_root, 'exp', 'rand_perm_cam.mat'))['rand_perm_cam']
            eval_sysu(q_feats, q_ids, q_cams, q_img_paths,
                      g_feats, g_ids, g_cams, g_img_paths,
                      perm, mode='all', num_shots=1, rerank=engine.rerank)
            eval_sysu(q_feats, q_ids, q_cams, q_img_paths,
                      g_feats, g_ids, g_cams, g_img_paths,
                      perm, mode='all', num_shots=10, rerank=engine.rerank)
            eval_sysu(q_feats, q_ids, q_cams, q_img_paths,
                      g_feats, g_ids, g_cams, g_img_paths,
                      perm, mode='indoor', num_shots=1, rerank=engine.rerank)
            eval_sysu(q_feats, q_ids, q_cams, q_img_paths,
                      g_feats, g_ids, g_cams, g_img_paths,
                      perm, mode='indoor', num_shots=10, rerank=engine.rerank)
        elif dataset == 'regdb':
            logger.info('Test Mode - infrared to visible')
            eval_regdb(q_feats, q_ids, q_cams, q_img_paths,
                       g_feats, g_ids, g_cams, g_img_paths, mode='i2v', rerank=engine.rerank)
            logger.info('Test Mode - visible to infrared')
            eval_regdb(g_feats, g_ids, g_cams, g_img_paths,
                       q_feats, q_ids, q_cams, q_img_paths, mode='v2i', rerank=engine.rerank)
        elif dataset == 'llcm':
            logger.info('Test Mode - infrared to visible')
            eval_llcm(
                q_feats, q_ids, q_cams, q_img_paths,
                g_feats, g_ids, g_cams, g_img_paths,
                mode='i2v',
                rerank=engine.rerank
            )
            logger.info('Test Mode - visible to infrared')
            eval_llcm(
                g_feats, g_ids, g_cams, g_img_paths,
                q_feats, q_ids, q_cams, q_img_paths,
                mode='v2i',
                rerank=engine.rerank
            )
        else:
            raise NotImplementedError(f'Dataset - {dataset} is not supported')

        evaluator.state.feat_list.clear()
        evaluator.state.id_list.clear()
        evaluator.state.cam_list.clear()
        evaluator.state.img_path_list.clear()
        del q_feats, q_ids, q_cams, g_feats, g_ids, g_cams

        logger.info('\n## Starting the training process...')

    @trainer.on(Events.STARTED)
    def train_start(engine):
        setattr(engine.state, "best_rank1", 0.0)
        #run_init_eval(engine)

    @trainer.on(Events.COMPLETED)
    def train_completed(engine):
        pass

    @trainer.on(Events.EPOCH_STARTED)
    def epoch_started_callback(engine):
        kv_metric.reset()
        timer.reset()

    @trainer.on(Events.EPOCH_COMPLETED)
    def epoch_completed_callback(engine):
        epoch = engine.state.epoch

        if lr_scheduler is not None:
            lr_scheduler.step()

        if epoch % eval_interval == 0:
            logger.info("Model saved at {}/{}_model_{}.pth".format(save_dir, prefix, epoch))

        if evaluator and epoch % eval_interval == 0 and epoch >= start_eval:
            # Extract Query Feature
            evaluator.run(query_loader)
            q_feats = torch.cat(evaluator.state.feat_list, dim=0)
            q_ids = torch.cat(evaluator.state.id_list, dim=0).numpy()
            q_cams = torch.cat(evaluator.state.cam_list, dim=0).numpy()
            q_img_paths = np.concatenate(evaluator.state.img_path_list, axis=0)

            # Extract Gallery Feature
            evaluator.run(gallery_loader)
            g_feats = torch.cat(evaluator.state.feat_list, dim=0)
            g_ids = torch.cat(evaluator.state.id_list, dim=0).numpy()
            g_cams = torch.cat(evaluator.state.cam_list, dim=0).numpy()
            g_img_paths = np.concatenate(evaluator.state.img_path_list, axis=0)

            if dataset == 'sysu':
                perm = sio.loadmat(os.path.join(dataset_cfg.sysu.data_root, 'exp', 'rand_perm_cam.mat'))[
                    'rand_perm_cam']
                r1, r5, r10, r20, mAP, mINP, _ = eval_sysu(q_feats, q_ids, q_cams, q_img_paths,
                                                           g_feats, g_ids, g_cams, g_img_paths,
                                                           perm, mode='all', num_shots=1, rerank=False)
            elif dataset == 'regdb':
                logger.info('Test Mode - infrared to visible')
                r1, r5, r10, r20, mAP, mINP, _ = eval_regdb(q_feats, q_ids, q_cams, q_img_paths,
                                                            g_feats, g_ids, g_cams, g_img_paths, mode='i2v')
                logger.info('Test Mode - visible to infrared')
                r1_, r5_, r10_, r20_, mAP_, mINP_, _ = eval_regdb(g_feats, g_ids, g_cams, g_img_paths,
                                                                  q_feats, q_ids, q_cams, q_img_paths, mode='v2i')
                r1 = (r1 + r1_) / 2
                r5 = (r5 + r5_) / 2
                r10 = (r10 + r10_) / 2
                r20 = (r20 + r20_) / 2
                mAP = (mAP + mAP_) / 2
                mINP = (mINP + mINP_) / 2
            elif dataset == 'llcm':
                logger.info('Test Mode - infrared to visible')
                r1, r5, r10, r20, mAP, mINP, _ = eval_llcm(
                    q_feats, q_ids, q_cams, q_img_paths,
                    g_feats, g_ids, g_cams, g_img_paths,
                    mode='i2v',
                    rerank=engine.rerank
                )

                logger.info('Test Mode - visible to infrared')
                r1_, r5_, r10_, r20_, mAP_, mINP_, _ = eval_llcm(
                    g_feats, g_ids, g_cams, g_img_paths,
                    q_feats, q_ids, q_cams, q_img_paths,
                    mode='v2i',
                    rerank=engine.rerank
                )

                r1 = (r1 + r1_) / 2
                r5 = (r5 + r5_) / 2
                r10 = (r10 + r10_) / 2
                r20 = (r20 + r20_) / 2
                mAP = (mAP + mAP_) / 2
                mINP = (mINP + mINP_) / 2
            else:
                raise NotImplementedError(f'Dataset - {dataset} is not supported')

            if r1 > engine.state.best_rank1:
                for rm_best_model_path in glob("{}/{}_model_best-*.pth".format(save_dir, prefix)):
                    os.remove(rm_best_model_path)
                engine.state.best_rank1 = r1
                torch.save(model.state_dict(), "{}/{}_model_best-{}.pth".format(save_dir, prefix, epoch))

            if writer is not None:
                writer.add_scalar('eval/r1', r1, epoch)
                writer.add_scalar('eval/r5', r5, epoch)
                writer.add_scalar('eval/r10', r10, epoch)
                writer.add_scalar('eval/r20', r20, epoch)
                writer.add_scalar('eval/mAP', mAP, epoch)
                writer.add_scalar('eval/mINP', mINP, epoch)

            evaluator.state.feat_list.clear()
            evaluator.state.id_list.clear()
            evaluator.state.cam_list.clear()
            evaluator.state.img_path_list.clear()
            del q_feats, q_ids, q_cams, g_feats, g_ids, g_cams

    @trainer.on(Events.ITERATION_COMPLETED)
    def iteration_complete_callback(engine):
        timer.step()
        kv_metric.update(engine.state.output)

        epoch = engine.state.epoch
        iteration = engine.state.iteration
        iter_in_epoch = iteration - (epoch - 1) * len(engine.state.dataloader)

        if iter_in_epoch % log_period == 0 and iter_in_epoch > 0:
            batch_size = engine.state.batch[0].size(0)
            speed = batch_size / timer.value()
            msg = "Epoch[%d] Batch [%d] Speed: %.2f samples/sec" % (epoch, iter_in_epoch, speed)
            metric_dict = kv_metric.compute()
            if logger is not None:
                for k in sorted(metric_dict.keys()):
                    msg += "  %s: %.4f" % (k, metric_dict[k])
                    if writer is not None:
                        writer.add_scalar('metric/{}'.format(k), metric_dict[k], iteration)
                logger.info(msg)
            kv_metric.reset()
            timer.reset()

    return trainer
