import numpy as np
import torch

from torch.cuda.amp import autocast
from torch.autograd import no_grad
from ignite.engine import Engine, Events


# =========================================================
# TRAIN ENGINE (Native AMP)
# =========================================================
def create_train_engine(
    model,
    optimizer,
    scaler=None,
    use_amp=False,
    non_blocking=False
):
    device = torch.device("cuda", torch.cuda.current_device())

    def _process_func(engine, batch):
        model.train()

        data, labels, cam_ids, img_paths, img_idxes = batch
        epoch = engine.state.epoch

        data = data.to(device, non_blocking=non_blocking)
        labels = labels.to(device, non_blocking=non_blocking)
        cam_ids = cam_ids.to(device, non_blocking=non_blocking)

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=use_amp):
            loss, metric = model(
                data,
                labels,
                cam_ids=cam_ids,
                img_paths=img_paths,
                epoch=epoch
            )

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        return metric

    return Engine(_process_func)


# =========================================================
# EVAL ENGINE (Native AMP)
# =========================================================
def create_eval_engine(
    model,
    non_blocking=False,
    use_amp=False
):
    device = torch.device("cuda", torch.cuda.current_device())

    def _process_func(engine, batch):
        model.eval()

        data, labels, cam_ids, img_paths = batch[:4]
        data = data.to(device, non_blocking=non_blocking)
        cam_ids = cam_ids.to(device, non_blocking=non_blocking)

        with no_grad():
            with autocast(enabled=use_amp):
                feat = model(data, cam_ids=cam_ids)

        return (
            feat.float().cpu(),
            labels,
            cam_ids.cpu(),
            np.array(img_paths)
        )

    engine = Engine(_process_func)

    # ---------------- Clear buffers ----------------
    @engine.on(Events.EPOCH_STARTED)
    def clear_data(engine):
        engine.state.feat_list = []
        engine.state.id_list = []
        engine.state.cam_list = []
        engine.state.img_path_list = []

    # ---------------- Store outputs ----------------
    @engine.on(Events.ITERATION_COMPLETED)
    def store_data(engine):
        feat, pid, camid, path = engine.state.output
        engine.state.feat_list.append(feat)
        engine.state.id_list.append(pid)
        engine.state.cam_list.append(camid)
        engine.state.img_path_list.append(path)

    return engine
