import os
import time
from multiprocessing import Process, Manager
from time import sleep

import pandas as pd
from tqdm import tqdm

from simdistserve.cxl_test import run_cxl_exp
from simdistserve.benchmarks.search_configs import get_distserve_configs, get_vllm_config
from simdistserve.constants import ModelTypes

# Restrict runtime to <= 32 CPU core.
# RunPod encounters problem when using `os.cpu_count()`
# to query the number of CPUs
MAX_CPU_COUNT = min(os.cpu_count() - 2, 32)


def main(
    num_node: int, num_gpu_per_node: int, model_type: ModelTypes,
    is_dist_high: bool = True,
    offload_type: str = None,
    memory_threshold: float = 1,
    gpu_memory_size: int = 32 * 1024 * 1024 * 1024,
    cxl_memory_size: int = 700 * 1024 * 1024 * 1024,
    local_memory_size: int = 64 * 1024 * 1024 * 1024,###########
    cxl_load_time_per_mb:float = 0.015,
    local_load_time_per_mb:float = 0.05,
    N: int = 300,
    backend: str = "distserve", 
):
    """
    :return result: dict that maps config to the best_per_gpu_rate (int)
    """
    if backend == "distserve":
        configs = get_distserve_configs(
            model_type, num_node, num_gpu_per_node, is_dist_high
        )
    elif backend == "vllm":
        configs = get_vllm_config(
            model_type, num_node * num_gpu_per_node
        )

    processes = []
    # Add a multiproc shared dict
    with Manager() as manager:
        result = manager.dict()
        pbar = tqdm(enumerate(configs), total=len(configs))
        for pid, config in pbar:
            proc = Process(
                target=run_cxl_exp,    ######### Modified ##########
                args=(
                    model_type, config,
                    offload_type,
                    memory_threshold,
                    gpu_memory_size,
                    cxl_memory_size,
                    local_memory_size,
                    cxl_load_time_per_mb,
                    local_load_time_per_mb,
                    N,
                    backend, 
                ),

            )
            if len(processes) >= MAX_CPU_COUNT:
                # Pop a process that has finished running
                found = False
                while not found:
                    for i in range(len(processes)):
                        if not processes[i].is_alive():
                            processes[i].join()
                            processes.pop(i)
                            found = True
                            pbar.update(1)
                            break
                    sleep(0.2)

            proc.start()
            processes.append(proc)
            pass
        for proc in processes:
            pbar.update(1)
            proc.join()
        result = dict(result)
        return result



if __name__ == '__main__':
    data = []
    model = "opt_13b"####, "opt_66b", "opt_175b"  # 测试不同模型规模
    model_type = ModelTypes.model_str_to_object(model)
    print("Starting experiments...")  # 首个检查点#######################
    ngpu = 2
    print(f"Testing with {ngpu} GPUs...")
    print(f"Running Distserve ...")
    start = time.perf_counter()
    result = main(
        num_node=1, 
        num_gpu_per_node=ngpu,
        model_type=f'facebook/opt-{model.split("_")[1]}',
        is_dist_high=True,
        offload_type='cxl',
        memory_threshold=1,
        gpu_memory_size=32 * 1024 * 1024 * 1024,
        #cxl_memory_size=0,
        cxl_memory_size=700 * 1024 * 1024 * 1024,
        #local_memory_size=64 * 1024 * 1024 * 1024,
        local_memory_size=0,
        cxl_load_time_per_mb=0.015,
        local_load_time_per_mb=0.05,
        N=300,
        backend='distserve'
    )
    end = time.perf_counter()
    
    data.append({
        "name": f"Distservecxl_{model}",
        "ngpu": ngpu,
        "duration": end - start,
        "throughput": result.get('throughput', 0),
        "total_tokens": result.get('total_tokens', 0),
        "total_time": result.get('total_time', 0),
        "offload_amount": result.get('offload_amount', 0), 
        "load_amount": result.get('load_amount', 0),       
        "max_gpu_usage": result.get('max_gpu_usage', 0)   
    })
    print(f"----------------------------Final result------------------------------")
    for EXP in data:
        print(f"name:{EXP['name']},ngpu:{EXP['ngpu']},duration:{EXP['duration']},throughput:{EXP['throughput']},total_tokens:{EXP['total_tokens']},total_time:{EXP['total_time']},offload_amount:{EXP['offload_amount']},load_amount:{EXP['load_amount']},max_gpu_usage:{EXP['max_gpu_usage']}")

    # for model in models:
    #     model_type = ModelTypes.model_str_to_object(model)
    #     print(f"\nTesting model: {model}")  # 模型循环检查点#####################
        
    #     for ngpu in 2#[2, 4, 8, 16, 32]:
    #         print(f"  Testing with {ngpu} GPUs...")  # GPU配置检查点#####################
    #         # Distserve local
    #         print(f"    Running Distserve local...")#######################
            
            # start = time.perf_counter()
            # result1 = main(
            #     num_node=1, 
            #     num_gpu_per_node=ngpu,
            #     model_type=f'facebook/opt-{model.split("_")[1]}',  # 直接构造正确格式
            #     is_dist_high=True,
            #     offload_type='local',
            #     memory_threshold=1,
            #     gpu_memory_size=32 * 1024 * 1024 * 1024,
            #     cxl_memory_size=0,
            #     local_memory_size=64 * 1024 * 1024 * 1024,
            #     cxl_load_time_per_mb=0.015,
            #     local_load_time_per_mb=0.05,
            #     N=300,
            #     backend='distserve'
            # )
            # end = time.perf_counter()
            # duration = end - start
            # data.append({
            #     "name": f"DistserveLocal_{model}",
            #     "ngpu": ngpu,
            #     "duration": duration,
            #     "throughput": result1.get('throughput', 0),
            #     "total_tokens": result1.get('total_tokens', 0),
            #     "total_time": result1.get('total_time', 0),
            #     "offload_amount": result1.get('offload_amount', 0), 
            #     "load_amount": result1.get('load_amount', 0),       
            #     "max_gpu_usage": result1.get('max_gpu_usage', 0)   
            # })

        #     # CXL
        #     start = time.perf_counter()
        #     result2 = main(
        #         num_node=1, 
        #         num_gpu_per_node=ngpu,
        #         model_type=f'facebook/opt-{model.split("_")[1]}',  # 直接构造正确格式
        #         is_dist_high=True,
        #         offload_type='cxl',
        #         memory_threshold=1,
        #         gpu_memory_size=32 * 1024 * 1024 * 1024,
        #         cxl_memory_size=700 * 1024 * 1024 * 1024,
        #         local_memory_size=0,
        #         cxl_load_time_per_mb=0.015,
        #         local_load_time_per_mb=0.05,
        #         N=300,
        #         backend='distserve'
        #     )
        #     end = time.perf_counter()#############
        #     data.append({
        #         "model": model,
        #         "system": "CxlDistInfer",
        #         "ngpu": ngpu,
        #         "duration": end - start,
        #         "throughput": result2.get('throughput', 0),
        #         "total_tokens": result2.get('total_tokens', 0),
        #         "total_time": result2.get('total_time', 0),
        #         "offload_amount": result2.get('offload_amount', 0),  
        #         "load_amount": result2.get('load_amount', 0),       
        #         "max_gpu_usage": result2.get('max_gpu_usage', 0)   
        #     })

        #     # vLLM local
        #     start = time.perf_counter()
        #     result3 = main(
        #         num_node=1, 
        #         num_gpu_per_node=ngpu,
        #         model_type=f'facebook/opt-{model.split("_")[1]}',  # 直接构造正确格式
        #         is_dist_high=True,
        #         offload_type='local',
        #         memory_threshold=1,
        #         gpu_memory_size=32 * 1024 * 1024 * 1024,
        #         cxl_memory_size=0,
        #         local_memory_size=64 * 1024 * 1024 * 1024,
        #         cxl_load_time_per_mb=0.015,
        #         local_load_time_per_mb=0.05,
        #         N=300,
        #         backend='vllm'
        #     )
        #     end = time.perf_counter()#############
        #     data.append({
        #         "model": model,
        #         "system": "vLLM_local",
        #         "ngpu": ngpu,
        #         "duration": end - start,
        #         "throughput": result3.get('throughput', 0),
        #         "total_tokens": result3.get('total_tokens', 0),
        #         "total_time": result3.get('total_time', 0),
        #         "offload_amount": result3.get('offload_amount', 0),  
        #         "load_amount": result3.get('load_amount', 0),       
        #         "max_gpu_usage": result3.get('max_gpu_usage', 0)   
        # })

    # 保存结果
    df = pd.DataFrame(data)
    df.to_csv("parallel_bisect_results.csv", index=False)

    pass
