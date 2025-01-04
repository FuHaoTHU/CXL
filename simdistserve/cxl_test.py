import argparse

import os
import time
from multiprocessing import Process, Manager
from time import sleep

import pandas as pd
from tqdm import tqdm

from simdistserve.benchmarks.search_configs import get_distserve_configs, get_vllm_config
from simdistserve.constants import ModelTypes
from simdistserve.benchmarks.simulate_cxl import run_experiment, parse_args
import json
from pathlib import Path

# Restrict runtime to <= 32 CPU core.
# RunPod encounters problem when using `os.cpu_count()`
# to query the number of CPUs
MAX_CPU_COUNT = min(os.cpu_count() - 2, 32)

def run_cxl_exp(
        model_type: ModelTypes,
        # config
        # - distserve: (pp_cross, tp_prefill, pp_prefill, tp_decode, pp_decode)
        # - vllm: (tp, pp)
        config,
        offload_type: str,########
        memory_threshold,
        gpu_memory_size,
        cxl_memory_size,
        local_memory_size,
        cxl_load_time_per_mb,
        local_load_time_per_mb,
        N: int = 300,#########
        backend: str = 'distserve',#########
):  
    print(f"CXL test starting with model: {model_type}, config: {config}")  
    N = str(N)

    if backend == 'distserve':
        print(f"Preparing distserve config...") 
        (pp_cross, tp_prefill, pp_prefill, tp_decode, pp_decode) = config
        num_gpu = pp_cross * (tp_prefill * pp_prefill + tp_decode * pp_decode)
        config_args = [
            '--tp-prefill', f'{tp_prefill}',
            '--pp-prefill', f'{pp_cross * pp_prefill}',
            '--tp-decode', f'{tp_decode}',
            '--pp-decode', f'{pp_cross * pp_decode}',
            '--offload_type', f'{offload_type}',
            '--memory-threshold', f'{memory_threshold}',  
            '--gpu-memory-size', f'{32 * 1024}',  # 32GB
            '--cxl-memory-size',  f'{700 * 1024}' if offload_type == 'cxl' else '0',  # 700GB for CXL
            '--local-memory-size', f'{64 * 1024}' if offload_type == 'local' else '0',  # 64GB for local
            '--cxl-load-time-per-mb', f'{cxl_load_time_per_mb}',
            '--local-load-time-per-mb', f'{local_load_time_per_mb}',
            '--N', f'{N}'
        ]
    else:
        (tp, pp) = config
        num_gpu = tp * pp
        config_args = [
            '--tp-prefill', f'{tp}',
            '--pp-prefill', f'{pp}',
            '--tp-decode', 0,
            '--pp-decode', 0,
            '--offload_type', f'{offload_type}',#########
            '--N', f'{N}'############
        ]
    
    fixed_args = [
        '--arrival', 'poisson',
        '--seed', '0',
        '--model', model_type,  
        '--workload', 'sharegpt',
        '--slas', '[]',
        '--slo-scales', '[1]',
        '--backend', backend,
    ]

    args = [*fixed_args, *config_args]
    args = [str(i) for i in args]
    args = parse_args(args)

    print(f"Running experiment with args: {args}")  

    
    print("\nStarting optimal threshold search")
    # if args.offload_type != 'cxl':
    #     print("Not CXL mode, returning default threshold")
   
    if args.offload_type == 'cxl':

        left, right = 0.3, 1.0  
        best_threshold = memory_threshold
        best_throughput = float('-inf')
        iteration = 0
        
        while (right - left) > 0.01:
            iteration += 1
            mid = (left + right) / 2
            args.memory_threshold = mid
            print(f"Iteration {iteration}, testing threshold: {mid}")
            STATS = run_experiment(args)
            print(f"Throughput at threshold {mid}: {STATS['throughput']}")
            if STATS['throughput'] > best_throughput:
                best_throughput = STATS['throughput']
                best_threshold = mid
                print(f"Best threshold updated to {best_threshold}")
                print(f"Best throughput updated to {best_throughput}")
                left = mid
            else:
                right = mid
        print(f"\nThreshold search completed:")
        print(f"- Best threshold: {best_threshold}")
        print(f"- Best throughput: {best_throughput}")
        STATS_FINAL = run_experiment(args)
        return STATS_FINAL

    else:
        print("Not CXL mode")
        STATS_FINAL = run_experiment(args)
        print(f"Throughput is {STATS_FINAL['throughput']}")
        return STATS_FINAL



# def find_optimal_threshold(self):
#     
#     print("\nStarting optimal threshold search")
#     if self.offload_type != 'cxl':
#         print("Not CXL mode, returning default threshold")
#         return self.memory_threshold

#     left, right = 0.3, 1.0  
#     best_threshold = self.memory_threshold
#     best_throughput = float('-inf')
#     current_TPOP = 0
#     iteration = 0

#     while right - left > 0.01:
#         iteration += 1
#         mid = (left + right) / 2
#         print(f"\nIteration {iteration}, testing threshold: {mid}")
#         current_throughput, TPOP = self.evaluate_threshold(mid)
#         print(f"Current threshold is {mid} and current throughput is {current_throughput}")
#         if current_throughput > best_throughput:
#             best_throughput = current_throughput
#             print(f"Best threshold updated to {best_threshold}")
#             current_TPOP = TPOP
#             best_threshold = mid
#             print(f"Found better threshold: {best_threshold}")
#             left = mid
#         else:
#             right = mid
#     print(f"\nThreshold search completed:")
#     print(f"- Best threshold: {best_threshold}")
#     print(f"- Best throughput: {best_throughput}")

#     return best_threshold, current_TPOP    #########Modified#########

# def evaluate_threshold(self, threshold):
# 

#     print(f"\nEvaluating threshold: {threshold}")
    
    

# 
#     self.memory_threshold = threshold
#     for worker in self.get_all_workers():
#         worker.memory_threshold = threshold
#         self.env.process(worker.run()) ###################
#         # worker.run()  

#     self.env.run(until=self.env.now + 1000)
        

#
#     performance, TPOP, Total_offload_amount, Total_load_amount = self.calculate_performance()
#     print(f"Evaluation results for threshold {threshold}:")
#     print(f"- Throughput: {performance}")
#     print(f"- TPOP: {TPOP}")
        
#     return performance, TPOP   ##############  To Be Modified ##############


# def calculate_performance(self):

# ä
# 
#     print(f"\nCalculating performance at simulation time: {self.env.now}")
#     total_tokens = 0
#     TPOP = 0
#     count = 0
#     total_offload_amount = 0
#     total_load_amount = 0
#     
#     decode_workers = set(worker for instance in self.decode_instances for worker in instance)
#     print("Checking all workers status:")
#     for worker in self.get_all_workers():
#     
#         print(f"Worker {worker.wid}:")
#         print(f"  - total_tokens: {worker.stats['total_tokens']}")
#         print(f"  - current time: {self.env.now}")
#         print(f"  - queue length: {len(worker.decode_queue)}")
#         total_tokens += worker.stats['total_tokens']
#         if worker in decode_workers:
#             count += 1
#             TPOP += worker.TPOP
#         else:######?
#             count = count
#             TPOP = TPOP
#         total_offload_amount += worker.stats['offload_amount']
#         total_load_amount += worker.stats['load_amount']
#     print(f"Performance calculation results:")
#     print(f"- Total tokens: {total_tokens}")
#     print(f"- Simulation time: {self.env.now}")
    
#     #if self.env.now == 0:
#     #   return float('-inf'),0,0,0
    
#     avg_TPOP = TPOP/count if count > 0 else 0
#     
#     throughput = total_tokens / self.env.now if self.env.now > 0 else 0
#     print(f"- Throughput: {throughput}")
#     print(f"- Average TPOP: {avg_TPOP}")
#     return throughput, avg_TPOP, total_offload_amount, total_load_amount  

def run_throughput_comparison():
    
    results = {}
    
    
    models = ['13b', '66b', '175b']
    systems = [
        ('distserve', 'local', (1,1,1,1,1)),  # Distserve_local
        ('distserve', 'cxl', (1,1,1,1,1)),    # CxlDistInfer
        ('vllm', 'local', (1,1))              # vLLM_local
    ]

    for model in models:
        print(f"\nTesting model: {model}")
        model_results = {}
        
        for backend, offload_type, config in systems:
            print(f"Testing system: {backend}-{offload_type}")
            
            stats = run_experiment(
                model_type=f'opt_{model}',
                config=config,
                backend=backend,
                offload_type=offload_type
            )
            
            if stats:
                model_results[f"{backend}_{offload_type}"] = {
                    'throughput': stats['throughput'],
                    'total_tokens': stats['total_tokens'],
                    'total_time': stats['total_time'],
                    'num_gpus': stats['num_gpus']
                }
        
        results[model] = model_results

    
    save_dir = Path("results")
    save_dir.mkdir(exist_ok=True)
    
    with open(save_dir / "throughput_comparison.json", 'w') as f:
        json.dump(results, f, indent=4)
        
    
    print("\nExperiment Results:")
    for model, model_results in results.items():
        print(f"\nModel: {model}")
        print(f"{'System':<20} {'Throughput (tokens/s)':<20}")
        print("-" * 40)
        for system, stats in model_results.items():
            print(f"{system:<20} {stats['throughput']:<20.2f}")


if __name__ == '__main__':
    args = parse_args()
    run_cxl_exp(
        ModelTypes.opt_13b,
        (1, 1, 1, 1, 1),
        "distserve",
        (200, 100, 90, 90),
        max_per_gpu_rate=16,
        is_debug=args.debug,
    )
            
