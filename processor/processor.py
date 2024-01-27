import logging
import os
import time
import torch
import torch.nn as nn
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval
from torch.cuda import amp
import torch.distributed as dist
from utils.visualizer import Visualizer
import pandas as pd
def do_train(cfg, model, center_criterion, train_loader, val_loader, optimizer, optimizer_center, scheduler, loss_fn, num_query, local_rank):
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD

    device = "cuda"
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger("Reid.train")
    logger.info('start training')
    _LOCAL_PROCESS_GROUP = None
    if device:
        model.to(local_rank)
        if torch.cuda.device_count() > 1 and cfg.MODEL.DIST_TRAIN:
            print('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM,reranking=False)
    scaler = amp.GradScaler()
    loss_list = []

    # train
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        loss_meter.reset()
        acc_meter.reset()
        evaluator.reset()
        scheduler.step(epoch)
        model.train()
        for n_iter, (img, vid, target_cam, target_view) in enumerate(train_loader):
            optimizer.zero_grad()
            optimizer_center.zero_grad()
            img = img.to(device)
            target = vid.to(device)
            with amp.autocast(enabled=True):
                score, feat = model(img)
                loss = loss_fn(score, feat, target)

            scaler.scale(loss).backward()

            scaler.step(optimizer)
            scaler.update()

            if 'center' in cfg.MODEL.METRIC_LOSS_TYPE:
                for param in center_criterion.parameters():
                    param.grad.data *= (1. / cfg.SOLVER.CENTER_LOSS_WEIGHT)
                scaler.step(optimizer_center)
                scaler.update()
            if isinstance(score, list):
                acc = (score[0].max(1)[1] == target).float().mean()
            else:
                acc = (score.max(1)[1] == target).float().mean()

            loss_meter.update(loss.item(), img.shape[0])
            acc_meter.update(acc, 1)

            torch.cuda.synchronize()
            if (n_iter + 1) % log_period == 0:
                logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                            .format(epoch, (n_iter + 1), len(train_loader),loss_meter.avg, acc_meter.avg, scheduler._get_lr(epoch)[0]))

        loss_list.append(loss_meter.avg)

        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)
        if cfg.MODEL.DIST_TRAIN:
            pass
        else:
            logger.info("Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                    .format(epoch, time_per_batch, train_loader.batch_size / time_per_batch))

        if epoch % checkpoint_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    torch.save(model.state_dict(),os.path.join(cfg.OUTPUT_DIR, 'ConformReid' + '_{}.pth'.format(epoch)))
            else:
                torch.save(model.state_dict(),os.path.join(cfg.OUTPUT_DIR, 'ConformReid' + '_{}.pth'.format(epoch)))

        if epoch % eval_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    model.eval()
                    for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(val_loader):
                        with torch.no_grad():
                            img = img.to(device)
                            feat = model(img)
                            evaluator.update((feat, vid, camid))
                    cmc, mAP, distmat, pids, camids, qf, gf, q_pids, g_pids, q_camids, g_camids, all_AP, all_INP = evaluator.compute()

                    logger.info("Validation Results - Epoch: {}".format(epoch))
                    logger.info("mAP: {:.1%}".format(mAP))
                    for r in [1, 5, 10]:
                        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                    torch.cuda.empty_cache()
            else:
                model.eval()
                for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(val_loader):
                    with torch.no_grad():
                        img = img.to(device)
                        feat = model(img)
                        evaluator.update((feat, vid, camid))
                cmc, mAP, distmat, pids, camids, qf, gf, q_pids, g_pids, q_camids, g_camids, all_AP, all_INP = evaluator.compute()

                logger.info("Validation Results - Epoch: {}".format(epoch))
                logger.info("mAP: {:.1%}".format(mAP))
                for r in [1, 5, 10]:
                    logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                torch.cuda.empty_cache()
        loss_dataset = pd.DataFrame(data = loss_list)
        loss_dataset.to_csv(os.path.join(cfg.OUTPUT_DIR, cfg.DATASETS.NAMES + '_loss' + '_{}.csv'.format(epoch)), encoding = 'gbk')

    # loss_dataset = pd.DataFrame(data = loss_list)
    # loss_dataset.to_csv(os.path.join(cfg.OUTPUT_DIR, cfg.DATASETS.NAMES + '_loss' + '_{}.csv'.format(epochs)), encoding = 'gbk')

def do_inference(cfg, model, val_loader, num_query):
    device = "cuda"
    logger = logging.getLogger("Reid.test")
    logger.info("Enter inferencing")

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM,reranking=False)

    evaluator.reset()

    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)

    model.eval()
    img_path_list = []

    for n_iter, (img, pid, camid, camids, target_view, imgpath) in enumerate(val_loader):
        with torch.no_grad():
            img = img.to(device)
            feat = model(img)
            evaluator.update((feat, pid, camid))
            img_path_list.extend(imgpath)

    cmc, mAP, distmat, pids, camids, qf, gf, q_pids, g_pids, q_camids, g_camids, all_AP, all_INP = evaluator.compute()
    logger.info("Validation Results ")
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))


    logger.info("Finish computing APs for all query images!")

    visualizer = Visualizer(val_loader.dataset)
    visualizer.get_model_output(all_AP, distmat, q_pids, g_pids, q_camids, g_camids)

    logger.info("Start saving ROC curve ...")
    fpr, tpr, pos, neg = visualizer.vis_roc_curve('./outputs/Visualization_rank_list')
    visualizer.save_roc_info('./outputs/Visualization_rank_list', fpr, tpr, pos, neg)
    logger.info("Finish saving ROC curve!")

    logger.info("Saving rank list result ...")
    query_indices = visualizer.vis_rank_list('./outputs/Visualization_rank_list', True, 100, 'ascending', 'ascending', 10)
    logger.info("Finish saving rank list results!")
