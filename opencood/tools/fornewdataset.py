'''
Prediction model input data generation.
SizheWei 2023.4.23
'''

import os
import argparse
import torch
from scipy import stats
from collections import OrderedDict

from opencood.hypes_yaml import yaml_utils
from opencood.data_utils.datasets import build_dataset


def prefer_json_path(yaml_path):
    json_path = yaml_path.replace("yaml", "json")
    json_path = json_path.replace(
        "OPV2V_irregular_npy", "OPV2V_irregular_npy_updated")
    return json_path if os.path.exists(json_path) else yaml_path

def retrieve_base_data(scenario_database, len_record, idx, binomial_n=10, binomial_p=0.1, k=3, is_no_shift=False, is_same_sample_interval=False):
    """
    Given the index, return the corresponding data.

    Parameters
    ----------
    idx : int
        Index given by dataloader.

    Returns
    -------
    data : dict
        The dictionary contains loaded yaml params and lidar data for
        each cav.
        Structure: 
        {
            cav_id_1 : {
                'ego' : true,
                curr : {
                    'params': json_path,
                    'timestamp': string
                },
                past_k : {		                # (k) totally
                    [0]:{
                        'params': json_path,
                        'timestamp': string,
                        'time_diff': float,
                        'sample_interval': int
                    },
                    [1] : {},	(id-1)
                    ...,		
                    [k-1] : {} (id-(k-1))
                },
            }, 
            cav_id_2 : {		                # (k) totally
                'ego': false, 
                ...
            }, 
            ...
        }
    """
    sample_interval_exp = int(binomial_n * binomial_p)
    # we loop the accumulated length list to get the scenario index
    scenario_index = 0
    for i, ele in enumerate(len_record):
        if idx < ele:
            scenario_index = i
            break
    scenario_database = scenario_database[scenario_index]
    
    # 生成冻结分布函数
    bernoulliDist = stats.bernoulli(binomial_p) 

    data = OrderedDict()
    # 找到 current 时刻的 timestamp_index 这对于每辆车来讲都一样
    curr_timestamp_idx = idx if scenario_index == 0 else \
                    idx - len_record[scenario_index - 1]
    curr_timestamp_idx = curr_timestamp_idx + binomial_n * k
    
    # load files for all CAVs
    for cav_id, cav_content in scenario_database.items():
        '''
        cav_content 
        {
            timestamp_1 : {
                yaml: path,
                lidar: path, 
                cameras: list of path
            },
            ...
            timestamp_n : {
                yaml: path,
                lidar: path, 
                cameras: list of path
            },
            'regular' : {
                timestamp_1 : {},
                ...
                timestamp_n : {}
            },
            'ego' : true / false , 
        },
        '''
        data[cav_id] = OrderedDict()
        
        # 1. set 'ego' flag
        data[cav_id]['ego'] = cav_content['ego']
        
        # 2. current frame, for co-perception lable use
        data[cav_id]['curr'] = {}

        timestamp_key = list(cav_content['regular'].items())[curr_timestamp_idx][0]
        
        # 2.1 load curr params
        # json is faster than yaml
        json_file = prefer_json_path(cav_content['regular'][timestamp_key]['yaml'])
        data[cav_id]['curr']['params'] = json_file

        # 2.3 store curr timestamp and time_diff
        data[cav_id]['curr']['timestamp'] = timestamp_key
        data[cav_id]['curr']['time_diff'] = 0.0
        data[cav_id]['curr']['sample_interval'] = 0

        # 3. past frames, for model input
        data[cav_id]['past_k'] = OrderedDict()
        latest_sample_stamp_idx = curr_timestamp_idx
        # past k frames, pose | lidar | label(for single view confidence map generator use)
        for i in range(k):
            # sample_interval
            if data[cav_id]['ego']:             # ego sample_interval = E(B(n, p))
                if i == 0: # ego-past-0 与 ego-curr 是一样的
                    data[cav_id]['past_k'][i] = data[cav_id]['curr']
                    continue
                sample_interval = sample_interval_exp
                if sample_interval == 0:
                    sample_interval = 1
            else:                               # non-ego sample_interval ~ B(n, p)
                if sample_interval_exp==0 \
                    and is_no_shift \
                        and i == 0:
                    data[cav_id]['past_k'][i] = data[cav_id]['curr']
                    continue
                if is_same_sample_interval:
                    sample_interval = sample_interval_exp
                else:
                    # B(n, p)
                    trails = bernoulliDist.rvs(binomial_n)
                    sample_interval = sum(trails)
                if sample_interval==0:
                    if i==0: # 检查past 0 的实际时间是否在curr 的后面
                        tmp_time_key = list(cav_content.items())[latest_sample_stamp_idx][0]
                        if dist_time(tmp_time_key, data[cav_id]['curr']['timestamp'])>0:
                            sample_interval = 1
                    if i>0: # 过去的几帧不要重复
                        sample_interval = 1                

            # check the timestamp index
            data[cav_id]['past_k'][i] = {}
            latest_sample_stamp_idx -= sample_interval
            timestamp_key = list(cav_content.items())[latest_sample_stamp_idx][0]
            # load the corresponding data into the dictionary
            # load param file: json is faster than yaml
            json_file = prefer_json_path(cav_content[timestamp_key]['yaml'])
            data[cav_id]['past_k'][i]['params'] = json_file

            data[cav_id]['past_k'][i]['timestamp'] = timestamp_key
            data[cav_id]['past_k'][i]['sample_interval'] = sample_interval
            data[cav_id]['past_k'][i]['time_diff'] = \
                dist_time(timestamp_key, data[cav_id]['curr']['timestamp'])

    return data

def dist_time(ts1, ts2, i = -1):
    """caculate the time interval between two timestamps

    Args:
        ts1 (string): time stamp at some time
        ts2 (string): current time stamp
        i (int, optional): past frame id, for debug use. Defaults to -1.
    
    Returns:
        time_diff (float): time interval (ts1 - ts2)
    """
    if not i==-1:
        return -i
    else:
        return (float(ts1) - float(ts2))

def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate CoBEVFlow Part-2 asynchronous sample specs.')
    parser.add_argument('--hypes_yaml', '-y', default='',
                        help='Yaml used to build IntermediateFusionDatasetIrregular.')
    parser.add_argument('--scenario_database', default='',
                        help='Optional prebuilt scenario_database.pt.')
    parser.add_argument('--len_record', default='',
                        help='Optional prebuilt len_record.pt.')
    parser.add_argument('--root_dir', default='',
                        help='Override root_dir when building from yaml.')
    parser.add_argument('--data_dir', default='',
                        help='Override data_dir/dair_data_dir when building from yaml.')
    parser.add_argument('--validate_dir', default='',
                        help='Override validate_dir when building from yaml.')
    parser.add_argument('--split', choices=['train', 'val'], default='train',
                        help='Dataset split to build when using --hypes_yaml.')
    parser.add_argument('--output_dir', required=True,
                        help='Directory to write scenario_database.pt, len_record.pt, and samples.')
    parser.add_argument('--output_name', default='part2_async_samples.pt',
                        help='Output filename for sampled specs.')
    parser.add_argument('--max_samples', type=int, default=0,
                        help='Maximum samples to export; 0 exports all samples.')
    parser.add_argument('--binomial_n', type=int, default=None)
    parser.add_argument('--binomial_p', type=float, default=None)
    parser.add_argument('--k', type=int, default=None)
    parser.add_argument('--is_no_shift', action='store_true')
    parser.add_argument('--is_same_sample_interval', action='store_true')
    parser.add_argument('--save_database', action='store_true',
                        help='Also save scenario_database.pt and len_record.pt.')
    return parser.parse_args()


def apply_path_overrides(hypes, args):
    if args.data_dir:
        hypes['data_dir'] = args.data_dir
        hypes['dair_data_dir'] = args.data_dir
    if args.root_dir:
        hypes['root_dir'] = args.root_dir
    if args.validate_dir:
        hypes['validate_dir'] = args.validate_dir
    return hypes


def resolve_sampling_args(hypes, args):
    return {
        'binomial_n': args.binomial_n
            if args.binomial_n is not None else hypes.get('binomial_n', 10),
        'binomial_p': args.binomial_p
            if args.binomial_p is not None else hypes.get('binomial_p', 0.1),
        'k': args.k if args.k is not None else hypes.get('num_sweep_frames', 3),
        'is_no_shift': args.is_no_shift or hypes.get('is_no_shift', False),
        'is_same_sample_interval': args.is_same_sample_interval
            or hypes.get('is_same_sample_interval', False),
    }


def load_sample_source(args):
    if args.scenario_database:
        if not args.len_record:
            raise ValueError('--len_record is required with --scenario_database.')
        scenario_database = torch.load(args.scenario_database)
        len_record = torch.load(args.len_record)
        hypes = {}
        return {
            'type': 'scenario_database',
            'scenario_database': scenario_database,
            'len_record': len_record,
            'total_samples': int(len_record[-1]),
        }, hypes

    if not args.hypes_yaml:
        raise ValueError('Either --hypes_yaml or --scenario_database is required.')

    hypes = yaml_utils.load_yaml(args.hypes_yaml)
    hypes = apply_path_overrides(hypes, args)
    dataset = build_dataset(
        hypes, visualize=False, train=(args.split == 'train'))
    if hasattr(dataset, 'scenario_database') and hasattr(dataset, 'len_record'):
        return {
            'type': 'scenario_database',
            'scenario_database': dataset.scenario_database,
            'len_record': dataset.len_record,
            'total_samples': int(dataset.len_record[-1]),
        }, hypes
    if hasattr(dataset, 'retrieve_async_sample_spec'):
        return {
            'type': 'dataset_exporter',
            'dataset': dataset,
            'total_samples': len(dataset),
        }, hypes
    raise AttributeError(
        'Dataset does not expose scenario_database/len_record or '
        'retrieve_async_sample_spec().')


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    sample_source, hypes = load_sample_source(args)
    sampling_args = resolve_sampling_args(hypes, args)

    if args.save_database and sample_source['type'] == 'scenario_database':
        scenario_database = sample_source['scenario_database']
        len_record = sample_source['len_record']
        torch.save(scenario_database,
                   os.path.join(args.output_dir, 'scenario_database.pt'))
        torch.save(len_record, os.path.join(args.output_dir, 'len_record.pt'))

    total_samples = sample_source['total_samples']
    export_samples = total_samples if args.max_samples <= 0 \
        else min(total_samples, args.max_samples)

    samples = []
    for idx in range(export_samples):
        if sample_source['type'] == 'scenario_database':
            samples.append(retrieve_base_data(
                sample_source['scenario_database'],
                sample_source['len_record'],
                idx,
                **sampling_args))
        else:
            samples.append(sample_source['dataset'].retrieve_async_sample_spec(idx))

    output_path = os.path.join(args.output_dir, args.output_name)
    torch.save({
        'samples': samples,
        'sampling_args': sampling_args,
        'num_samples': export_samples,
        'total_samples': total_samples,
    }, output_path)
    print('Saved %d/%d async sample specs to %s' %
          (export_samples, total_samples, output_path))


if __name__ == '__main__':
    main()
