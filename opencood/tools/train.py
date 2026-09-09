# -*- coding: utf-8 -*-
# Author: Runsheng Xu
# Modified: Sizhe Wei

import argparse
import os
import statistics
import sys
sys.path.append(os.getcwd())
import time

import torch
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

import importlib
import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils
from opencood.tools.diagnostics_utils import DiagnosticsManager
from opencood.data_utils.datasets import build_dataset

from tqdm import tqdm
from tqdm.contrib import tenumerate
from tqdm.auto import trange

import subprocess

run_test = True
# from opencood.data_utils.datasets.intermediate_fusion_dataset_opv2v_irregular import illegal_path_list
compensation = True 

def train_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument("--hypes_yaml", "-y", type=str, required=True,
                        help='data generation yaml file needed ')
    parser.add_argument('--model_dir', default='',
                        help='Continued training path')
    parser.add_argument('--fusion_method', '-f', default="intermediate",
                        help='passed to inference.')
    parser.add_argument('--pretrained_path', default='',
                        help='The path of the model need to be fine tuned.')
    parser.add_argument('--device', '-d', default="cuda", help='cuda or cpu')
    parser.add_argument('--two_stage', help='whether to use two stage training', default=0, type=int)
    parser.add_argument('--num_workers', default=16, type=int,
                        help='number of dataloader workers')
    parser.add_argument('--skip_test', action='store_true',
                        help='skip automatic inference after training')
    parser.add_argument('--debug_max_iters', default=0, type=int,
                        help='maximum training iterations per epoch for smoke tests')
    parser.add_argument('--debug_max_val_iters', default=0, type=int,
                        help='maximum validation iterations per epoch for smoke tests')
    parser.add_argument('--debug_epoches', default=0, type=int,
                        help='override train_params.epoches for smoke tests')
    parser.add_argument('--debug_batch_size', default=0, type=int,
                        help='override train_params.batch_size for smoke tests')
    parser.add_argument('--debug_max_cav', default=0, type=int,
                        help='override train_params.max_cav for smoke tests')
    parser.add_argument('--root_dir', default='',
                        help='override dataset root_dir')
    parser.add_argument('--data_dir', default='',
                        help='override dataset data_dir/dair_data_dir')
    parser.add_argument('--validate_dir', default='',
                        help='override dataset validate_dir')
    parser.add_argument('--test_dir', default='',
                        help='override dataset test_dir')
    parser.add_argument('--debug_regular_time', action='store_true',
                        help='use regular OPV2V timestamps for irregular dataset smoke tests')
    parser.add_argument('--debug_disable_single_supervise', action='store_true',
                        help='disable single-view auxiliary supervision for smoke tests')
    parser.add_argument('--debug_unfreeze_model', action='store_true',
                        help='disable model backbone_fix/only_tune_header flags for smoke tests')
    parser.add_argument('--diagnostics', action='store_true',
                        help='enable reusable scalar/visual diagnostics')
    parser.add_argument('--diagnostics_interval', default=0, type=int,
                        help='override diagnostics scalar logging interval')
    parser.add_argument('--diagnostics_grad_interval', default=0, type=int,
                        help='override diagnostics gradient norm interval')
    parser.add_argument('--diagnostics_val_vis_interval', default=0, type=int,
                        help='save one validation BEV visualization every N batches')
    parser.add_argument('--diagnostics_max_val_vis', default=0, type=int,
                        help='maximum validation visualizations saved by diagnostics')
    opt = parser.parse_args()
    return opt


def apply_debug_overrides(hypes, opt):
    if opt.data_dir:
        hypes['data_dir'] = opt.data_dir
        hypes['dair_data_dir'] = opt.data_dir
    if opt.root_dir:
        hypes['root_dir'] = opt.root_dir
    if opt.validate_dir:
        hypes['validate_dir'] = opt.validate_dir
    if opt.test_dir:
        hypes['test_dir'] = opt.test_dir
    if opt.debug_epoches > 0:
        hypes['train_params']['epoches'] = opt.debug_epoches
    if opt.debug_batch_size > 0:
        hypes['train_params']['batch_size'] = opt.debug_batch_size
    if opt.debug_max_cav > 0:
        hypes['train_params']['max_cav'] = opt.debug_max_cav
        if 'model' in hypes and 'args' in hypes['model']:
            hypes['model']['args']['max_cav'] = opt.debug_max_cav
    if opt.debug_regular_time:
        hypes['is_ab_regular'] = True
    if opt.debug_disable_single_supervise:
        hypes['train_params']['supervise_single_flag'] = False
    if opt.debug_unfreeze_model and 'model' in hypes and 'args' in hypes['model']:
        hypes['model']['args']['backbone_fix'] = False
        hypes['model']['args']['only_tune_header'] = False
    if opt.diagnostics:
        hypes.setdefault('diagnostics', {})
        hypes['diagnostics']['enabled'] = True
    if opt.diagnostics_interval > 0:
        hypes.setdefault('diagnostics', {})
        hypes['diagnostics']['scalar_interval'] = opt.diagnostics_interval
    if opt.diagnostics_grad_interval > 0:
        hypes.setdefault('diagnostics', {})
        hypes['diagnostics']['grad_interval'] = opt.diagnostics_grad_interval
    if opt.diagnostics_val_vis_interval > 0:
        hypes.setdefault('diagnostics', {})
        hypes['diagnostics'].setdefault('val_visualize', {})
        hypes['diagnostics']['val_visualize']['enabled'] = True
        hypes['diagnostics']['val_visualize']['interval'] = opt.diagnostics_val_vis_interval
    if opt.diagnostics_max_val_vis > 0:
        hypes.setdefault('diagnostics', {})
        hypes['diagnostics'].setdefault('val_visualize', {})
        hypes['diagnostics']['val_visualize']['max_images'] = opt.diagnostics_max_val_vis
    return hypes


def add_auxiliary_losses(hypes, output_dict, final_loss):
    aux_weights = hypes.get('loss', {}).get('args', {}).get('aux_weights', {})
    if not aux_weights:
        return final_loss

    for loss_name, weight in aux_weights.items():
        if loss_name not in output_dict:
            continue
        aux_loss = output_dict[loss_name]
        if torch.is_tensor(aux_loss) and aux_loss.numel() == 1:
            final_loss = final_loss + float(weight) * aux_loss
    return final_loss


def main():
    opt = train_parser()
    if str(opt.device).lower() == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    elif str(opt.device).isdigit():
        os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.device)

    hypes = yaml_utils.load_yaml(opt.hypes_yaml, opt)
    hypes = apply_debug_overrides(hypes, opt)

    finetune_flag = False
    if 'is_finetune' in hypes:
        finetune_flag = hypes['is_finetune']
    if finetune_flag:
        finetune_time_interval = int(hypes['binomial_n'] * hypes['binomial_p'])
        print('### Finetune mode, only tune header. ###')

    print('### Dataset Building ... ###')
    start_time = time.time()
    opencood_train_dataset = build_dataset(hypes, visualize=False, train=True)
    opencood_validate_dataset = build_dataset(hypes,
                                              visualize=False,
                                              train=False)

    train_loader = DataLoader(opencood_train_dataset,
                            batch_size=hypes['train_params']['batch_size'],
                            num_workers=opt.num_workers,
                            collate_fn=opencood_train_dataset.collate_batch_train,
                            shuffle=True,
                            pin_memory=True,
                            drop_last=True)
    val_loader = DataLoader(opencood_validate_dataset,
                            batch_size=hypes['train_params']['batch_size'],
                            num_workers=opt.num_workers,
                            collate_fn=opencood_train_dataset.collate_batch_train,
                            shuffle=True,
                            pin_memory=True,
                            drop_last=True)
    end_time = time.time()
    print("=== Time consumed: %.1f minutes. ===" % ((end_time - start_time)/60))
    start_time = time.time()
    
    print('### Creating Model ... ###')
    model = train_utils.create_model(hypes)
    device = torch.device('cuda' if torch.cuda.is_available() and str(opt.device).lower() != "cpu" else 'cpu')
    print("device: ", device)

    # record lowest validation loss checkpoint.
    lowest_val_loss = 1e5
    lowest_val_epoch = -1    
    
    # load pre-train model for single view
    is_pre_trained = False
    if 'is_single_pre_trained' in hypes and hypes['is_single_pre_trained']['pre_train_flag']:
        is_pre_trained = True
    if is_pre_trained:
        pretrain_path = hypes['is_single_pre_trained']['pre_train_path']
        initial_epoch = hypes['is_single_pre_trained']['pre_train_epoch']
        pre_train_model = torch.load(os.path.join(pretrain_path, 'net_epoch%d.pth' % initial_epoch))
        model.load_state_dict(pre_train_model, strict=False)
        print("### Pre-trained point pillar {} loaded successfully! ###".format(os.path.join(pretrain_path, 'net_epoch%d.pth' % initial_epoch)))
        fix = hypes['is_single_pre_trained']['pre_train_fix']
        if fix:
            for name, value in model.named_parameters():
                if name in pre_train_model:
                    value.requires_grad = False

    # we assume gpu is necessary
    if torch.cuda.is_available():
        model.to(device)

    # define the loss 
    criterion = train_utils.create_loss(hypes)

    # optimizer setup
    optimizer = train_utils.setup_optimizer(hypes, model, is_pre_trained)
    
    # for fine tune:
    if opt.pretrained_path:
        saved_path = opt.pretrained_path
        pretrained_model_dict = torch.load(saved_path, map_location='cpu')
        diff_keys = {k:v for k, v in pretrained_model_dict.items() if k not in model.state_dict()}
        if diff_keys:
            print(f"!!! PreTrained model has keys: {diff_keys.keys()}, \
                which are not in the model you have created!!!")
        diff_keys = {k:v for k, v in model.state_dict().items() if k not in pretrained_model_dict.keys()}
        if diff_keys:
            print(f"!!! Created model has keys: {diff_keys.keys()}, \
                which are not in the model you have trained!!!")
        model.load_state_dict(pretrained_model_dict, strict=False)
        for name, value in model.named_parameters():
            if name in pretrained_model_dict:
                value.requires_grad = False
    
    # lr scheduler setup
    # if we want to train from last checkpoint.
    if opt.model_dir:
        saved_path = opt.model_dir
        init_epoch, model = train_utils.load_saved_model_diff(saved_path, model, finetune_flag)
        if finetune_flag:
            init_epoch = 0 # if finetune, we set the init_epoch to 10
        scheduler = train_utils.setup_lr_schedular(hypes, optimizer, init_epoch=init_epoch)

    else:
        init_epoch = 0
        log_path = '../logs' # Your log file path
        # if we train the model from scratch, we need to create a folder to save the model
        saved_path = train_utils.setup_train(hypes, log_path)
        scheduler = train_utils.setup_lr_schedular(hypes, optimizer)

    end_time = time.time()
    print("=== Time consumed: %.1f minutes. ===" % ((end_time - start_time)/60))

    # record training
    writer = SummaryWriter(saved_path)
    diagnostics = DiagnosticsManager(hypes, saved_path)

    start_time = time.time()
    print('### Training start! ###')
    epoches = hypes['train_params']['epoches']
    supervise_single_flag = False
    if 'supervise_single_flag' in hypes['train_params'].keys():
        supervise_single_flag = hypes['train_params']['supervise_single_flag']
        print(f"=== supervise_single_flag: {supervise_single_flag} ===")

    ############ For DiscoNet ##############
    if "kd_flag" in hypes.keys():
        kd_flag = True
        teacher_model_name = hypes['kd_flag']['teacher_model'] # point_pillar_disconet_teacher
        teacher_model_config = hypes['kd_flag']['teacher_model_config']
        teacher_checkpoint_path = hypes['kd_flag']['teacher_path']

        # import the model
        model_filename = "opencood.models." + teacher_model_name
        model_lib = importlib.import_module(model_filename)
        teacher_model_class = None
        target_model_name = teacher_model_name.replace('_', '')

        for name, cls in model_lib.__dict__.items():
            if name.lower() == target_model_name.lower():
                teacher_model_class = cls
        
        teacher_model = teacher_model_class(teacher_model_config)
        teacher_model.load_state_dict(torch.load(teacher_checkpoint_path), strict=False)
        
        for p in teacher_model.parameters():
            p.requires_grad_(False)

        if torch.cuda.is_available():
                teacher_model.to(device)

        teacher_model.eval()

    else:
        kd_flag = False

    sample_interval_all_epoch = 0
    for epoch in range(init_epoch, max(epoches, init_epoch)):
        for param_group in optimizer.param_groups:
            print('learning rate %f' % param_group["lr"])
        
        sample_interval = 0
        i = 0
        train_steps = 0
        for i, batch_data in enumerate(train_loader):
            if opt.debug_max_iters > 0 and train_steps >= opt.debug_max_iters:
                break
            if batch_data is None:
                continue
            model.train()
            model.zero_grad()
            optimizer.zero_grad()
            batch_data = train_utils.to_device(batch_data, device)

            batch_data['ego']['epoch'] = epoch

            # TODO: dataset parameter is only used for training flow module
            if opt.two_stage:
                ouput_dict = model(batch_data['ego'], opencood_train_dataset)
            else: 
                ouput_dict = model(batch_data['ego'])

            if kd_flag:
                teacher_output_dict = teacher_model(batch_data['ego'])
                ouput_dict.update(teacher_output_dict)

            # # only for SyncNet training
            # final_loss = criterion(ouput_dict, batch_data['ego']['label_dict'])
            # if i % 10 == 0:
            #     criterion.logging(epoch, i, len(train_loader), writer)

            # first argument is always your output dictionary,
            # second argument is always your label dictionary.
            
            if 'with_compensation' in hypes['model']['args'] and \
                hypes['model']['args']['with_compensation']:
                if hypes['model']['args']['with_single_supervise']:
                    final_loss = ouput_dict['recon_loss'] 
                    single_det_loss = criterion(ouput_dict, batch_data['ego']['single_object_label'], mode = 'single')
                    detection_loss = criterion(ouput_dict, batch_data['ego']['label_dict'])
                    final_loss = ouput_dict['recon_loss'] + single_det_loss + detection_loss
                else:
                    detection_loss = criterion(ouput_dict, batch_data['ego']['label_dict'])
                    final_loss = ouput_dict['recon_loss'] + detection_loss
            else:
                final_loss = criterion(ouput_dict, batch_data['ego']['label_dict'])
                if i % 10 == 0:
                    criterion.logging(epoch, i, len(train_loader), writer)
                if supervise_single_flag:
                    final_loss += criterion(ouput_dict, batch_data['ego']['single_object_label'], suffix="_single")
                    if i % 10 == 0:
                        criterion.logging(epoch, i, len(train_loader), writer, suffix="_single")
            final_loss = add_auxiliary_losses(hypes, ouput_dict, final_loss)

            # back-propagation
            final_loss.backward()
            diagnostics.log_train_step(
                writer, epoch, i, len(train_loader), final_loss,
                ouput_dict, optimizer)
            diagnostics.log_gradients(
                writer, model, epoch, i, len(train_loader))
            optimizer.step()
            train_steps += 1
        
        sample_interval /= max(train_steps, 1)
        sample_interval_all_epoch += sample_interval

        if epoch % hypes['train_params']['eval_freq'] == 0:
            valid_ave_loss = []
            end_time = time.time()
            print('### %d th epoch trained, start validation! Time consumed %.2f ###' % (epoch, (end_time - start_time)/60))
            with torch.no_grad():
                val_steps = 0
                for i, batch_data in tenumerate(val_loader):
                    if opt.debug_max_val_iters > 0 and val_steps >= opt.debug_max_val_iters:
                        break
                    if batch_data is None:
                        continue
                    model.zero_grad()
                    optimizer.zero_grad()
                    model.eval()

                    batch_data = train_utils.to_device(batch_data, device)
                    batch_data['ego']['epoch'] = epoch
                    # TODO: dataset parameter is only used for training flow module
                    if opt.two_stage:
                        ouput_dict = model(batch_data['ego'], opencood_validate_dataset)
                    else:
                        ouput_dict = model(batch_data['ego'])

                    if kd_flag:
                        teacher_output_dict = teacher_model(batch_data['ego'])
                        ouput_dict.update(teacher_output_dict)

                    final_loss = criterion(ouput_dict,
                                           batch_data['ego']['label_dict'])
                    final_loss = add_auxiliary_losses(hypes, ouput_dict, final_loss)
                    valid_ave_loss.append(final_loss.item())
                    diagnostics.log_validation_step(
                        writer, epoch, i, len(val_loader), final_loss,
                        ouput_dict)
                    diagnostics.maybe_save_validation_visual(
                        batch_data, ouput_dict, opencood_validate_dataset,
                        hypes, epoch, i)
                    val_steps += 1

            valid_ave_loss = statistics.mean(valid_ave_loss) if valid_ave_loss else float('inf')
            print('At epoch %d, the validation loss is %f' % (epoch,
                                                              valid_ave_loss))
            writer.add_scalar('Validate_Loss', valid_ave_loss, epoch)

            # lowest val loss
            if valid_ave_loss < lowest_val_loss:
                lowest_val_loss = valid_ave_loss
                save_checkpoint_prefix = 'net_epoch_bestval_at'
                if finetune_flag:
                    save_checkpoint_prefix = f'finetune_{finetune_time_interval}_' + save_checkpoint_prefix
                torch.save(model.state_dict(),
                       os.path.join(saved_path,
                                    f'{save_checkpoint_prefix}%d.pth' % (epoch + 1)))
                if lowest_val_epoch != -1 and os.path.exists(os.path.join(saved_path,
                                    f'{save_checkpoint_prefix}%d.pth' % (lowest_val_epoch))):
                        os.remove(os.path.join(saved_path,
                                    f'{save_checkpoint_prefix}%d.pth' % (lowest_val_epoch)))
                lowest_val_epoch = epoch + 1

        if epoch % hypes['train_params']['save_freq'] == 0:
            torch.save(model.state_dict(),
                       os.path.join(saved_path,
                                    'net_epoch%d.pth' % (epoch + 1)))
        scheduler.step(epoch)

    end_time = time.time()
    print("Time consumed: %.1f" % ((end_time - start_time)/60))
    sample_interval_all_epoch /= max(epoches, init_epoch) - init_epoch
    print("Avg sample interval of all epochs: %.2f" % sample_interval_all_epoch) 
    print('Training Finished, checkpoints saved to %s' % saved_path)
    torch.cuda.empty_cache()
    
    if run_test and not opt.skip_test:
        fusion_method = opt.fusion_method
        cmd = f"python opencood/tools/inference_multi_sweep.py --model_dir {saved_path} --fusion_method {fusion_method}"
        print(f"Running command: {cmd}")
        os.system(cmd)

if __name__ == '__main__':
    main()
