#!/usr/bin/env python3
""" ImageNet Validation Script

This is intended to be a lean and easily modifiable ImageNet validation script for evaluating pretrained
models or training checkpoints against ImageNet or similarly organized image datasets. It prioritizes
canonical PyTorch, standard Python style, and good performance. Repurpose as you see fit.

Hacked together by Ross Wightman (https://github.com/rwightman)
"""

""" GroupNL: Low-Resource and Robust CNN Design over Cloud and Device
ImageNet-C Validation
Modified from timm
Author: Xie Jianhang
Github: https://github.com/jianhayes
Email: jianhang.xie@my.cityu.edu.hk
"""

import argparse
import os
import csv
import glob
import time
import logging
import torch
import torch.nn as nn
import torch.nn.parallel
from collections import OrderedDict
from contextlib import suppress

import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable as V
import numpy as np

from timm.models import create_model, apply_test_time_pool, load_checkpoint, is_model, list_models
from timm.data import create_dataset, create_loader, resolve_data_config, RealLabelsImagenet
from timm.utils import accuracy, AverageMeter, natural_key, setup_default_logging, set_jit_legacy

import models


has_apex = False
try:
    from apex import amp
    has_apex = True
except ImportError:
    pass

has_native_amp = False
try:
    if getattr(torch.cuda.amp, 'autocast') is not None:
        has_native_amp = True
except AttributeError:
    pass


# raw INFO logger
def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


torch.backends.cudnn.benchmark = True
# _logger = logging.getLogger('validate')

parser = argparse.ArgumentParser(description='PyTorch ImageNet Validation')
parser.add_argument('--rename', type=str, default='r1')
parser.add_argument("--data-dir", type=str, default='./dataset/ImageNet-C/raw/')
parser.add_argument("--output-dir", type=str, default='./ddp_imgc')
parser.add_argument('--output', type=str, default='./ddp')
parser.add_argument('--model', '-m', metavar='NAME', default='./model_list.txt', help='model architecture (default: dpn92)')
parser.add_argument('-b', '--batch-size', default=256, type=int, metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--pretrained', dest='pretrained', action='store_true', help='use pre-trained model')
parser.add_argument('--amp', action='store_true', default=False, help='Use AMP mixed precision. Defaults to Apex, fallback to native Torch AMP.')
parser.add_argument('--apex-amp', action='store_true', default=False, help='Use NVIDIA Apex AMP mixed precision')
parser.add_argument('--native-amp', action='store_true', default=False, help='Use Native Torch AMP mixed precision')
parser.add_argument('--results-file', default='', type=str, metavar='FILENAME', help='Output csv file for validation results (summary)')
parser.add_argument('--checkpoint', default=None, type=str, metavar='PATH', help='path to latest checkpoint (default: none)')

parser.add_argument('--dataset', '-d', metavar='NAME', default='', help='dataset type (default: ImageFolder/ImageTar if empty)')
parser.add_argument('--split', metavar='NAME', default='validation', help='dataset split (default: validation)')
parser.add_argument('--dataset-download', action='store_true', default=False,
                    help='Allow download of dataset for torch/ and tfds/ datasets that support it.')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 2)')
parser.add_argument('--img-size', default=None, type=int, metavar='N', help='Input image dimension, uses model default if empty')
parser.add_argument('--input-size', default=None, nargs=3, type=int,
                    metavar='N N N', help='Input all image dimensions (d h w, e.g. --input-size 3 224 224), uses model default if empty')
parser.add_argument('--crop-pct', default=None, type=float, metavar='N', help='Input image center crop pct')
parser.add_argument('--mean', type=float, nargs='+', default=None, metavar='MEAN', help='Override mean pixel value of dataset')
parser.add_argument('--std', type=float,  nargs='+', default=None, metavar='STD', help='Override std deviation of of dataset')
parser.add_argument('--interpolation', default='', type=str, metavar='NAME', help='Image resize interpolation type (overrides model)')
parser.add_argument('--num-classes', type=int, default=None, help='Number classes in dataset')
parser.add_argument('--class-map', default='', type=str, metavar='FILENAME', help='path to class to idx mapping file (default: "")')
parser.add_argument('--gp', default=None, type=str, metavar='POOL',
                    help='Global pool type, one of (fast, avg, max, avgmax, avgmaxc). Model default if None.')
parser.add_argument('--log-freq', default=10, type=int, metavar='N', help='batch logging frequency (default: 10)')
parser.add_argument('--num-gpu', type=int, default=1, help='Number of GPUS to use')
parser.add_argument('--test-pool', dest='test_pool', action='store_true', help='enable test time pool')
parser.add_argument('--no-prefetcher', action='store_true', default=False, help='disable fast prefetcher')
parser.add_argument('--pin-mem', action='store_true', default=False, help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
parser.add_argument('--channels-last', action='store_true', default=False, help='Use channels_last memory layout')
parser.add_argument('--tf-preprocessing', action='store_true', default=False,
                    help='Use Tensorflow preprocessing pipeline (require CPU TF installed')
parser.add_argument('--use-ema', dest='use_ema', action='store_true', help='use ema version of weights if present')
parser.add_argument('--torchscript', dest='torchscript', action='store_true', help='convert model torchscript for inference')
parser.add_argument('--legacy-jit', dest='legacy_jit', action='store_true', help='use legacy jit mode for pytorch 1.5/1.5.1/1.6 to get back fusion performance')
parser.add_argument('--real-labels', default='', type=str, metavar='FILENAME', help='Real labels JSON file for imagenet evaluation')
parser.add_argument('--valid-labels', default='', type=str, metavar='FILENAME', help='Valid label indices txt file for validation of partial label space')
# test model_best or last
parser.add_argument('--best', action='store_true', default=False, help='Use best model for test (otherwise use last)')

log_args = parser.parse_args()
if log_args.best:
    str_log = 'model_best'
else:
    str_log = 'last'
log_dir = (log_args.output_dir if log_args.output_dir else './output/val_imagenet_c') + '/{}_val_imagenet_c_{}.log'.format(log_args.model, str_log)
_logger = get_logger(log_dir)

def auc(errs):  # area under the distortion-error curve
    area = 0
    for i in range(1, len(errs)):
        area += (errs[i] + errs[i - 1]) / 2
    area /= len(errs) - 1
    return area


def validate_imagenet_c(args):
    # might as well try to validate something
    args.pretrained = args.pretrained or not args.checkpoint
    args.prefetcher = not args.no_prefetcher
    amp_autocast = suppress  # do nothing
    if args.amp:
        if has_native_amp:
            args.native_amp = True
        elif has_apex:
            args.apex_amp = True
        else:
            _logger.warning("Neither APEX or Native Torch AMP is available.")
    assert not args.apex_amp or not args.native_amp, "Only one AMP mode should be set."
    if args.native_amp:
        amp_autocast = torch.cuda.amp.autocast
        _logger.info('Validating in mixed precision with native PyTorch AMP.')
    elif args.apex_amp:
        _logger.info('Validating in mixed precision with NVIDIA APEX AMP.')
    else:
        _logger.info('Validating in float32. AMP not enabled.')

    if args.legacy_jit:
        set_jit_legacy()

    # create model
    model = create_model(
        args.model,
        pretrained=args.pretrained,
        num_classes=args.num_classes,
        in_chans=3,
        global_pool=args.gp,
        scriptable=args.torchscript,
    )
    if args.num_classes is None:
        assert hasattr(model, 'num_classes'), 'Model must have `num_classes` attr if not set on cmd line/config.'
        args.num_classes = model.num_classes
    if not args.pretrained and args.checkpoint:
        print(args.checkpoint)
        load_checkpoint(model, args.checkpoint, args.use_ema)

    param_count = sum([m.numel() for m in model.parameters()])
    _logger.info('Model %s created, param count: %d' % (args.model, param_count))

    data_config = resolve_data_config(vars(args), model=model, use_test_size=True, verbose=True)
    test_time_pool = False
    if args.test_pool:
        model, test_time_pool = apply_test_time_pool(model, data_config, use_test_size=True)

    if args.torchscript:
        torch.jit.optimized_execution(True)
        model = torch.jit.script(model)

    model = model.cuda()
    if args.apex_amp:
        model = amp.initialize(model, opt_level='O1')

    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)

    if args.num_gpu > 1:
        model = torch.nn.DataParallel(model, device_ids=list(range(args.num_gpu)))

    # ImageNet-C testing
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    @torch.no_grad()
    def show_performance(distortion_name, model_, args):
        model_.eval()
        errs = []

        for severity in range(1, 6):
            distorted_dataset = datasets.ImageFolder(
                root=args.data_dir + distortion_name + '/' + str(severity),
                transform=transforms.Compose([transforms.CenterCrop(224),
                                              transforms.ToTensor(),
                                              transforms.Normalize(mean, std)]))

            distorted_dataset_loader = DataLoader(
                distorted_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=args.pin_mem)

            correct = 0
            for batch_idx, (data, target) in enumerate(distorted_dataset_loader):
                data = V(data.cuda())

                output = model_(data)

                pred = output.data.max(1)[1]
                correct += pred.eq(target.cuda()).sum()

            err = 1 - 1. * correct / len(distorted_dataset)
            errs.append(err.cpu().numpy())
            _logger.info('{} severity {} error (%): {:.3f}'.format(distortion_name, severity, 100 * err))

        # _logger.info('\n=Average', tuple(errs))
        return np.mean(errs)


    _logger.info('Using ImageNet-C data')

    distortions = [
        'gaussian_noise', 'shot_noise', 'impulse_noise',
        'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
        'snow', 'frost', 'fog', 'brightness',
        'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression',
        'speckle_noise', 'gaussian_blur', 'spatter', 'saturate'
    ]

    error_rates = []
    for distortion_name in distortions:
        rate = show_performance(distortion_name, model, args)
        error_rates.append(rate)
        _logger.info('Distortion: {:15s}  | CE (unnormalized) (%): {:.3f}'.format(distortion_name, 100 * rate))

    _logger.info('mCE (unnormalized errors) (%): {:.3f}'.format(100 * np.mean(error_rates)))
    _logger.info('mC Acc (%): {:.3f}'.format(100 - 100 * np.mean(error_rates)))

    results = OrderedDict(
        top1=round(100 - 100 * np.mean(error_rates), 4), top1_err=round(100 * np.mean(error_rates), 4),
        # top5=round(top5a, 4), top5_err=round(100 - top5a, 4),
        param_count=round(param_count / 1e6, 2),
        img_size=data_config['input_size'][-1],
        # cropt_pct=crop_pct,
        interpolation=data_config['interpolation'])

    return results


def main():
    setup_default_logging()
    args = parser.parse_args()
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)
    args.results_file = '{}/{}_imagenet-c_results_{}.csv'.format(args.output_dir, args.model, str_log)

    print("   ### Loading evaluation dataset from {} ###    ".format(args.data_dir))

    model_cfgs = []
    model_names = []
    if args.best:
        str_file = 'model_best.pth.tar'
    else:
        str_file = 'last.pth.tar'
    if args.checkpoint is not None:
        if os.path.isdir(args.checkpoint):
            args.checkpoint = '{}/{}'.format(args.checkpoint, str_file)
            # checkpoints = args.checkpoint
            checkpoints = glob.glob(args.checkpoint)
            print("   ### checkpoints is {} ###    ".format(checkpoints))
            model_names = list_models(args.model)
            model_cfgs = [(args.model, c) for c in sorted(checkpoints, key=natural_key)]
            print("   ### model_cfgs is {} ###    ".format(model_cfgs))
    else:
        if args.model == 'all':
            # validate all models in a list of names with pretrained checkpoints
            args.pretrained = True
            model_names = list_models(pretrained=True, exclude_filters=['*_in21k', '*_in22k'])
            model_cfgs = [(n, '') for n in model_names]
        elif not is_model(args.model):
            # model name doesn't exist, try as wildcard filter
            model_names = list_models(args.model)
            model_cfgs = [(n, '') for n in model_names]

        if not model_cfgs and os.path.isfile(args.model):
            with open(args.model) as f:
                model_names = [line.rstrip() for line in f]
            model_cfgs = [(n, None) for n in model_names if n]

    checkpoint = '{}/model_name_ddp_{}/{}'.format(args.output, args.rename, str_file)
    print(checkpoint)
    if len(model_cfgs):
        results_file = args.results_file
        _logger.info('Running bulk validation on these pretrained models: {}'.format(', '.join(model_names)))
        results = []
        try:
            start_batch_size = args.batch_size
            for m, c in model_cfgs:
                batch_size = start_batch_size
                args.model = m
                if args.checkpoint is None or c is None:
                    args.checkpoint = checkpoint.replace('model_name', args.model)
                    print('args.checkpoint is {}'.format(args.checkpoint))
                result = OrderedDict(model=args.model)
                r = {}
                while not r and batch_size >= args.num_gpu:
                    torch.cuda.empty_cache()
                    try:
                        args.batch_size = batch_size
                        print('Validating with batch size: %d' % args.batch_size)
                        r = validate_imagenet_c(args)
                    except RuntimeError as e:
                        if batch_size <= args.num_gpu:
                            print("Validation failed with no ability to reduce batch size. Exiting.")
                            raise e
                        batch_size = max(batch_size // 2, args.num_gpu)
                        print("Validation failed, reducing batch size by 50%")
                result.update(r)
                if args.checkpoint:
                    result['checkpoint'] = args.checkpoint
                results.append(result)
        except KeyboardInterrupt as e:
            pass
        results = sorted(results, key=lambda x: x['top1'], reverse=True)
        if len(results):
            write_results(results_file, results)
    else:
        validate_imagenet_c(args)


def write_results(results_file, results):
    with open(results_file, mode='a') as cf:
        dw = csv.DictWriter(cf, fieldnames=results[0].keys())
        dw.writeheader()
        for r in results:
            dw.writerow(r)
        cf.flush()


if __name__ == '__main__':
    main()



