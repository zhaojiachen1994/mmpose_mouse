import argparse
import copy
import os
import os.path as osp
import time
import warnings

import mmcv
import torch
import torch.distributed as dist
from mmcv import Config, DictAction
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (DistSamplerSeedHook, OptimizerHook, get_dist_info)
from mmcv.runner import init_dist, set_random_seed, IterBasedRunner
from mmcv.utils import digit_version
from mmcv.utils import get_git_hash
from torch.utils.data import Dataset

from mmpose import __version__
from mmpose.apis import init_random_seed
from mmpose.core import DistEvalHook, EvalHook, build_optimizers
from mmpose.core.distributed_wrapper import DistributedDataParallelWrapper
from mmpose.datasets import build_dataloader, build_dataset
from mmpose.models import build_posenet
from mmpose.utils import collect_env, setup_multi_processes
from mmpose.utils import get_root_logger

try:
    from mmcv.runner import Fp16OptimizerHook
except ImportError:
    warnings.warn(
        'Fp16OptimizerHook from mmpose will be deprecated from '
        'v0.15.0. Please install mmcv>=1.1.4', DeprecationWarning)
    from mmpose.core import Fp16OptimizerHook


def parse_args():
    parser = argparse.ArgumentParser(description='Train a pose model')
    parser.add_argument('--config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume-from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='(Deprecated, please use --gpu-id) number of gpus to use '
             '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='(Deprecated, please use --gpu-id) ids of gpus to use '
             '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='id of gpu to use '
             '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--diff_seed',
        action='store_true',
        help='Whether or not set different seeds for different ranks')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        default={},
        help='override some settings in the used config, the key-value pair '
             'in xxx=yyy format will be merged into config file. For example, '
             "'--cfg-options model.backbone.depth=18 model.backbone.with_cp=True'")
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument(
        '--autoscale-lr',
        action='store_true',
        help='automatically scale lr with the number of gpus')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def train_model(model,
                dataset,
                cfg,
                distributed=False,
                validate=False,
                timestamp=None,
                meta=None):
    """Train model entry function.

    Args:
        model (nn.Module): The model to be trained.
        dataset (Dataset): Train dataset.
        cfg (dict): The config dict for training.
        distributed (bool): Whether to use distributed training.
            Default: False.
        validate (bool): Whether to do evaluation. Default: False.
        timestamp (str | None): Local time for runner. Default: None.
        meta (dict | None): Meta dict to record some important information.
            Default: None
    """
    logger = get_root_logger(cfg.log_level)

    # prepare data loaders
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]
    # step 1: give default values and override (if exist) from cfg.data
    loader_cfg = {
        **dict(
            seed=cfg.get('seed'),
            drop_last=False,
            dist=distributed,
            num_gpus=len(cfg.gpu_ids)),
        **({} if torch.__version__ != 'parrots' else dict(
            prefetch_num=2,
            pin_memory=False,
        )),
        **dict((k, cfg.data[k]) for k in [
            'samples_per_gpu',
            'workers_per_gpu',
            'shuffle',
            'seed',
            'drop_last',
            'prefetch_num',
            'pin_memory',
            'persistent_workers',
        ] if k in cfg.data)
    }

    # step 2: cfg.data.train_dataloader has highest priority
    train_loader_cfg = dict(loader_cfg, **cfg.data.get('train_dataloader', {}))

    data_loaders = [build_dataloader(ds, **train_loader_cfg) for ds in dataset]

    # determine whether use adversarial training precess or not
    use_adverserial_train = cfg.get('use_adversarial_train', False)
    # use_adverserial_train = True

    # put model on gpus
    if distributed:
        find_unused_parameters = cfg.get('find_unused_parameters', True)
        # Sets the `find_unused_parameters` parameter in
        # torch.nn.parallel.DistributedDataParallel

        if use_adverserial_train:
            # Use DistributedDataParallelWrapper for adversarial training
            model = DistributedDataParallelWrapper(
                model,
                device_ids=[torch.cuda.current_device()],
                broadcast_buffers=False,
                find_unused_parameters=find_unused_parameters)
        else:
            model = MMDistributedDataParallel(
                model.cuda(),
                device_ids=[torch.cuda.current_device()],
                broadcast_buffers=False,
                find_unused_parameters=find_unused_parameters)
    else:
        if digit_version(mmcv.__version__) >= digit_version(
                '1.4.4') or torch.cuda.is_available():
            model = MMDataParallel(model, device_ids=cfg.gpu_ids)
        else:
            warnings.warn(
                'We recommend to use MMCV >= 1.4.4 for CPU training. '
                'See https://github.com/open-mmlab/mmpose/pull/1157 for '
                'details.')

    # build optimizer
    optimizer = build_optimizers(model, cfg.optimizer)
    # optimizer=None

    # runner = EpochBasedRunner(
    #     model,
    #     optimizer=optimizer,
    #     work_dir=cfg.work_dir,
    #     logger=logger,
    #     meta=meta)

    runner = IterBasedRunner(
        model,
        optimizer=optimizer,
        work_dir=cfg.work_dir,
        logger=logger,
        meta=meta

    )

    # an ugly workaround to make .log and .log.json filenames the same
    runner.timestamp = timestamp

    if use_adverserial_train:
        # The optimizer step process is included in the train_step function
        # of the model, so the runner should NOT include optimizer hook.
        optimizer_config = None
    else:
        # fp16 setting
        fp16_cfg = cfg.get('fp16', None)
        if fp16_cfg is not None:
            optimizer_config = Fp16OptimizerHook(
                **cfg.optimizer_config, **fp16_cfg, distributed=distributed)
        elif distributed and 'type' not in cfg.optimizer_config:
            optimizer_config = OptimizerHook(**cfg.optimizer_config)
        else:
            optimizer_config = cfg.optimizer_config

    custom_hooks_cfg = cfg.get('custom_hooks', None)
    if custom_hooks_cfg is None:
        custom_hooks_cfg = cfg.get('custom_hooks_config', None)
        if custom_hooks_cfg is not None:
            warnings.warn(
                '"custom_hooks_config" is deprecated, please use '
                '"custom_hooks" instead.', DeprecationWarning)

    # register hooks
    runner.register_training_hooks(
        cfg.lr_config,
        optimizer_config,
        cfg.checkpoint_config,
        cfg.log_config,
        cfg.get('momentum_config', None),
        custom_hooks_config=custom_hooks_cfg)

    if distributed:
        runner.register_hook(DistSamplerSeedHook())

    # register eval hooks
    if validate:
        eval_cfg = cfg.get('evaluation', {})
        val_dataset = build_dataset(cfg.data.val, dict(test_mode=True))
        dataloader_setting = dict(
            samples_per_gpu=1,
            workers_per_gpu=cfg.data.get('workers_per_gpu', 1),
            # cfg.gpus will be ignored if distributed
            num_gpus=len(cfg.gpu_ids),
            dist=distributed,
            drop_last=False,
            shuffle=False)
        dataloader_setting = dict(dataloader_setting,
                                  **cfg.data.get('val_dataloader', {}))
        val_dataloader = build_dataloader(val_dataset, **dataloader_setting)
        eval_hook = DistEvalHook if distributed else EvalHook
        runner.register_hook(eval_hook(val_dataloader, **eval_cfg))

    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)
    runner.run(data_loaders, cfg.workflow, cfg.total_iters)


def main(args):
    cfg = Config.fromfile(args.config)

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # set multi-process settings
    setup_multi_processes(cfg)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    if args.gpus is not None:
        cfg.gpu_ids = range(1)
        warnings.warn('`--gpus` is deprecated because we only support '
                      'single GPU mode in non-distributed training. '
                      'Use `gpus=1` now.')
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids[0:1]
        warnings.warn('`--gpu-ids` is deprecated, please use `--gpu-id`. '
                      'Because we only support single GPU mode in '
                      'non-distributed training. Use the first GPU '
                      'in `gpu_ids` now.')
    if args.gpus is None and args.gpu_ids is None:
        cfg.gpu_ids = [args.gpu_id]

    if args.autoscale_lr:
        # apply the linear scaling rule (https://arxiv.org/abs/1706.02677)
        cfg.optimizer['lr'] = cfg.optimizer['lr'] * len(cfg.gpu_ids) / 8

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
        if len(cfg.gpu_ids) > 1:
            warnings.warn(
                f'We treat {cfg.gpu_ids} as gpu-ids, and reset to '
                f'{cfg.gpu_ids[0:1]} as gpu-ids to avoid potential error in '
                'non-distribute training time.')
            cfg.gpu_ids = cfg.gpu_ids[0:1]
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
        # re-set gpu_ids with distributed training mode
        _, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)

    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)
    meta['env_info'] = env_info

    # log some basic info
    logger.info(f'Distributed training: {distributed}')
    logger.info(f'Config:\n{cfg.pretty_text}')

    # set random seeds
    seed = init_random_seed(args.seed)
    seed = seed + dist.get_rank() if args.diff_seed else seed
    logger.info(f'Set random seed to {seed}, '
                f'deterministic: {args.deterministic}')
    set_random_seed(seed, deterministic=args.deterministic)
    cfg.seed = seed
    meta['seed'] = seed

    model = build_posenet(cfg.model)

    # todo: add the source dataset and target dataset
    datasets = [build_dataset(cfg.data.train)]

    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        val_dataset.pipeline = cfg.data.train.pipeline
        datasets.append(build_dataset(val_dataset))

    if cfg.checkpoint_config is not None:
        # save mmpose version, config file content
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            mmpose_version=__version__ + get_git_hash(digits=7),
            config=cfg.pretty_text,
        )
    train_model(
        model,
        datasets,
        cfg,
        distributed=distributed,
        # validate=(not args.no_validate),
        validate=True,
        timestamp=timestamp,
        meta=meta)


if __name__ == '__main__':
    args = parse_args()
    # args.config = "../configs/mouse/try_adv_train.py"
    args.config = "../configs/mouse/try_em_train.py"
    args.gpu_id = 0
    main(args)
