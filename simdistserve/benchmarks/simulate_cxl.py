"""
Simulate DistServe CXL 

Output a JSON (list) where each item is the lifecycle for a request.
"""
import argparse
import json
import os
import random
from pathlib import Path
from typing import Literal, Union
import time  # ############
import numpy as np
import pandas as pd
import simpy

from simdistserve.base.organize_data import organize_request_df, organize_request_event_df, \
    calculate_per_request_latency, organize_worker_event_df
from simdistserve.base.scheduler import put_requests_with_interarrivals
from simdistserve.base.worker import WorkerConfig
from simdistserve.base.workload import (
    get_gamma_interarrival,
    get_fixed_interarrival,
    convert_absolutearrival_to_interarrival, convert_pd_pair_to_request, sample_requests
)
from simdistserve.clusters.disagg import DisaggCluster
from simdistserve.clusters.vllm import VLLMCluster
from simdistserve.constants import ModelTypes
from simdistserve.estimators.memory_estimator import get_max_num_tokens, is_model_runnable

def parse_args(args_=None):
    parser = argparse.ArgumentParser(description='Simulation: vLLM, DistServe')
    parser.add_argument('--backend', type=str, default='distserve',
                        help='Backend to simulate (distserve, vllm)')
    parser.add_argument('--model', type=str, default='facebook/opt-13b',
                        help='Model type (opt_13b, opt_66b, opt_175b,'
                             'or facebook/opt-13b, facebook/opt-66b, facebook/opt-175b)')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--rate', type=float, default=float("inf"),
                        help='Rate of requests per second')
    parser.add_argument('--N', type=int, default=64, help='Number of requests')
    parser.add_argument(
        '--arrival', type=str, default='poisson',
        help=('Arrival distribution (gamma, poisson, fixed, custom). '
              'If custom, then require the JSON file workload to specify '
              'the "start_time" field for each incoming request.'))
    parser.add_argument(
        '--workload', type=str, default='sharegpt',
        help=(
            'Workload type, or a JSON file that contains the workload. '
            'The workload file should be a list of pairs with (prompt_len, decode_len) length. '
            '(e.g.: "sharegpt", "longbench", "humaneval", or specify your own path like "./workload/workload.json")')
    )
    parser.add_argument('--cv', type=float, default=1.0)
    parser.add_argument('--tp-prefill', type=int, default=1, help='Number of TP per prefill worker (used in DistServe)')
    parser.add_argument('--pp-prefill', type=int, default=1, help='Number of PP per prefill worker (used in DistServe)')
    parser.add_argument('--tp-decode', type=int, default=1, help='Number of TP per decode worker (used in DistServe)')
    parser.add_argument('--pp-decode', type=int, default=1, help='Number of PP per decode worker (used in DistServe)')
    parser.add_argument('--name', type=str, default=None)  # Experiment name
    parser.add_argument('--output', type=str, default=None, help='Output SLA (csv)')
    parser.add_argument('--output-request-info', type=str, default=None, help='Output request info (csv)')
    parser.add_argument('--output-request-event', type=str, default=None,
                        help='Output per-request event dataframe (csv)')
    parser.add_argument('--output-request-latency', type=str, default=None, help='Output per-request latency (csv)')
    parser.add_argument('--output-worker', type=str, default=None,
                        help='Output per-worker per-iteration time (csv)')
    parser.add_argument('--prefill-containment', type=int, default=None,
                        help='Containment target for prefill')
    parser.add_argument('--prefill-target', type=int, default=200,
                        help='Target latency for prefill')
    parser.add_argument('--decode-containment', type=int, default=None,
                        help='Containment target for decode')
    parser.add_argument('--slas', type=str, default='[85, 90, 95, 98, 99]',
                        help='Fix attainment and get the target.'),
    parser.add_argument('--slo-scales', type=str, default='[1.0, 0.4, 0.6, 0.8, 1.2]',
                        help='SLO scales in a python list.'),
    parser.add_argument('--decode-target', type=int, default=100,
                        help='Target latency for decode')
    parser.add_argument('--verbose', action='store_true', default=False,
                        help='Print verbose output')
    parser.add_argument('--offload_type', type=str, default=None,
                        help='Offload type: cxl, local, None')
    parser.add_argument('--memory-threshold', type=float, default=0.8,
                        help='Memory threshold')
    parser.add_argument('--gpu-memory-size', type=int, default=32 * 1024 * 1024 * 1024,
                        help='GPU memory size')
    parser.add_argument('--cxl-memory-size', type=int, default=700 * 1024 * 1024 * 1024,
                        help='CXL memory size')
    parser.add_argument('--local-memory-size', type=int, default=0,
                        help='Local memory size')
    parser.add_argument('--cxl-load-time-per-mb', type=float, default=0.015,
                        help='CXL load time per MB')
    parser.add_argument('--local-load-time-per-mb', type=float, default=0.05,
                        help='Local load time per MB')

    args = parser.parse_args(args=args_)

    assert args.backend in ['distserve', 'vllm'], f'Unknown backend: {args.backend}'
    assert args.arrival in ['poisson', 'gamma', 'fixed', 'custom'], f'Unknown arrival process: {args.arrival}'
    args.slo_scales = eval(args.slo_scales)
    args.slas = eval(args.slas)
    assert isinstance(args.slo_scales, list)
    assert isinstance(args.slas, list)
    return args


def check_dataset_existence(x):
    if not Path(x).exists():
        raise FileNotFoundError(f"Dataset {x} does not exist.")
    return


def load_workload(workload, N, rate, cv, seed, process: Literal["fixed", "gamma"]):
    random.seed(seed)
    np.random.seed(seed)
    if workload in ['sharegpt', 'longbench', 'humaneval']:
        dataset_root = os.environ.get('DATASET', '/workspace/CXL/Distserve/dataset')
        dataset_root = Path(dataset_root)
        assert dataset_root.exists(), (
            f"Dataset root {dataset_root} does not exist. "
            f"Please set the env var `DATASET` to the correct path."
        )
        dataset_file = dataset_root / f"{workload}.ds"
        check_dataset_existence(dataset_file)
        requests = sample_requests(dataset_file, N)

        if process == 'fixed':
            delay = 1 / rate * 1000  # ms
            arrival = get_fixed_interarrival(N, delay)
        else:
            arrival = get_gamma_interarrival(N, rate, cv, seed=seed)

    else:
        # Open the file to get the JSON data
        # [ { "start_time": int, "prompt_len": int, "output_len":int,  } ]
        with open(workload, 'r') as f:
            data = json.load(f)
        request_pairs = [(d['prompt_len'], d['output_len']) for d in data]
        requests = convert_pd_pair_to_request(request_pairs)
        absolute_arrival = [d['start_time'] for d in data]
        arrival = convert_absolutearrival_to_interarrival(absolute_arrival)
        pass
    return requests, arrival

def main(args, outputs = None):
    print("Starting main simulation...")
    outputs = outputs if outputs is not None else {}
    cv = args.cv
    N = args.N
    rate = args.rate
    seed = args.seed
    workload: Union[Literal["sharegpt", "longbench", "humaneval"], str] = args.workload
    model_type = args.model
    process = args.arrival
    print(f"Loading workload with N={args.N}, rate={args.rate}, process={args.arrival}")
    print(f"Checking model configuration...")
    TP_Prefill = args.tp_prefill
    PP_prefill = args.pp_prefill
    TP_Decode = args.tp_decode
    PP_decode = args.pp_decode
    num_gpus = TP_Prefill * PP_prefill + TP_Decode * PP_decode  #GPU number
    print(f"GPU configuration: {num_gpus} GPUs total")
    # 在这里处理模型名称转换###########################
    if args.model.startswith('opt_'):
        # 如果是 opt_13b 格式，转换为 facebook/opt-13b 格式
        model_type = f'facebook/opt-{args.model.split("_")[1]}'
    else:
        # 如果已经是 facebook/opt-13b 格式，直接使用
        model_type = args.model
        
    print(f"Model type: {model_type}")  # 添加调试输出
    
    process = args.arrival
    print(f"Loading workload with N={args.N}, rate={args.rate}, process={args.arrival}")
    ####################################################
    # Handle vllm in data processing
    #
    if not is_model_runnable(model_type, TP_Prefill, PP_prefill):
        raise ValueError(
            f"Model {model_type} is not runnable with TP={TP_Prefill}, PP={PP_prefill}. "
            f"Skipping by throwing exception..."
        )
    
    prefill_max_tokens = get_max_num_tokens(model_type, TP_Prefill, PP_prefill)
    if args.backend == 'vllm':
        TP_Decode = PP_decode = 0
        decode_max_tokens = prefill_max_tokens
        pass
    else:
        decode_max_tokens = get_max_num_tokens(model_type, TP_Decode, PP_decode)

    print(f"Loading dataset from {args.workload}...")
    # Setting the seed to sample request / process
    requests, arrival = load_workload(workload, N, rate, cv, seed, process)
    print(f"Loading dataset from {args.workload}...")


    print(f"Creating simulation environment...")
    env = simpy.Environment()
    print(f"Creating cluster with backend: {args.backend}")
    if args.backend == 'vllm':
        worker_config = WorkerConfig(
            model_type=model_type,
            TP=TP_Prefill, TP_Prefill=TP_Prefill, TP_Decode=TP_Prefill,
            prefill_max_batch_size=10 ** 7,  # inf
            decode_max_batch_size=10 ** 7,  # inf
            prefill_max_tokens=prefill_max_tokens,
            decode_max_tokens=decode_max_tokens,
            enable_chunked_prefill=False,
            engine_type=args.backend,
        )

        cluster = VLLMCluster(
            env=env, PP=PP_prefill, worker_configs=worker_config,
        )
    elif args.backend == 'distserve':
        print("Configuring distserve worker...")
        worker_config = WorkerConfig(
            model_type=model_type,
            TP=TP_Prefill, TP_Prefill=TP_Prefill, TP_Decode=TP_Decode,
            prefill_max_batch_size=10 ** 7,  # inf
            decode_max_batch_size=10 ** 7,  # inf
            prefill_max_tokens=prefill_max_tokens,
            decode_max_tokens=decode_max_tokens,
            enable_chunked_prefill=False,
            engine_type=args.backend, 
        )
        
        print("Creating DisaggCluster...")

        cluster = DisaggCluster(
            env=env, PP_prefill=PP_prefill, PP_decode=PP_decode,
            worker_configs=worker_config,
            offload_type=args.offload_type,
            memory_threshold=args.memory_threshold,
            gpu_memory_size=args.gpu_memory_size,
            cxl_memory_size=args.cxl_memory_size,
            local_memory_size=args.local_memory_size,
            cxl_load_time_per_mb=args.cxl_load_time_per_mb,
            local_load_time_per_mb=args.local_load_time_per_mb,
        )
    else:
        raise ValueError(f"Unknown backend: {args.backend}")
    print("Resetting worker stats...")
    for worker in cluster.get_all_workers():
        worker.reset_stats()
    print("Starting cluster...")
    cluster.run()
    print("Submitting requests...")
    put_requests_with_interarrivals(env, cluster.scheduler, arrival, requests)
    print("Running simulation...")
    start_time = time.time()  # 添加开始时间记录
    print("Before simulation, checking state:")
    print(f"- Number of requests: {len(requests)}")
    print(f"- Worker count: {len(cluster.get_all_workers())}")
    print(f"- Request arrival intervals: {arrival[:5]}...")  # 只打印前5个间隔
    env.run()
    print("After simulation, checking state:")
    print(f"- Simulation time: {env.now}")
    end_time = time.time()    # 添加结束时间记录
    print("Simulation completed, collecting stats...")

        # 收集性能指标
    total_tokens = 0
    total_time = 0
    offload_amount = 0
    load_amount = 0
    max_gpu_usage = 0
    

    
    print("Collecting worker stats...")
# 从所有worker收集统计信息
    for i, worker in enumerate(cluster.get_all_workers()):
        print(f"\nWorker {i} full stats dictionary:")
        print(worker.stats)  # 打印整个 stats 字典
        try:

            total_tokens += worker.stats['total_tokens']
            print(f"Successfully got total_tokens: {worker.stats['total_tokens']}")
            #total_time += worker.stats['total_delay']
            if args.offload_type == 'cxl':
                total_time += worker.stats['load_amount'] * 10**(-6) * 0.0078125 + worker.stats['total_delay']
            elif args.offload_type == 'local':
                total_time += worker.stats['load_amount'] * 10**(-6) * 0.03125 + worker.stats['total_delay']
            else:
                total_time += worker.stats['total_delay']
            print(f"Successfully got total_delay: {worker.stats['total_delay']}")
            offload_amount += worker.stats['offload_amount']
            print(f"Successfully got offload_amount: {worker.stats['offload_amount']}")
            load_amount += worker.stats['load_amount']
            print(f"Successfully got load_amount: {worker.stats['load_amount']}")
            max_gpu_usage = max(max_gpu_usage, worker.stats['max_gpu_memory_usage'])
            print(f"Successfully got max_gpu_usage: {worker.stats['max_gpu_memory_usage']}")
        except KeyError as e:
            print(f"KeyError when accessing {e} in worker {i}")
            print("Available keys in stats:", list(worker.stats.keys()))

    

    # 计算吞吐量和其他指标
    stats = {
        'throughput': total_tokens / total_time if total_time > 0 else 0,  # tokens/s
        'num_gpus': num_gpus,
        'total_tokens': total_tokens,
        'total_time': total_time,
        'offload_amount': offload_amount,
        'load_amount': load_amount,
        'max_gpu_usage': max_gpu_usage
    }

    print(f"Throughput: {stats['throughput']}")
    print(f"Total Delay: {stats['total_time']}")
    return stats

run_experiment = main
