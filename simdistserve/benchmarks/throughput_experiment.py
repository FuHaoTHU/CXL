import simpy
import numpy as np
from typing import List, Dict
import matplotlib.pyplot as plt
from pathlib import Path

from simdistserve.clusters.vllm import VLLMCluster
from simdistserve.base.scheduler import put_requests_with_interarrivals
from simdistserve.structs import Dataset, TestRequest

def run_single_experiment(
    model_size: str,
    generation_length: int,
    engine_type: str = "distserve",
    offload_type: str = None,
    num_requests: int = 100,
    simulation_time: int = 1000,
):
    """运行单次实验"""
    env = simpy.Environment()
    
    # 配置worker参数
    worker_configs = {
        "model_type": f"opt_{model_size}",
        "engine_type": engine_type,
        "offload_type": offload_type,
        "prefill_max_tokens": 2048,
        "decode_max_tokens": generation_length,
    }
    
    # 创建集群
    cluster = VLLMCluster(
        env=env,
        N_instance=1,
        PP=1,
        model_size=model_size,
        engine_type=engine_type,
        offload_type=offload_type,
        worker_configs=worker_configs
    )
    
    # 生成请求
    requests = []
    for i in range(num_requests):
        req = TestRequest(
            prompt="test prompt",
            prompt_len=128,  # 固定prompt长度
            output_len=generation_length
        )
        requests.append(req)
    
    # 设置请求到达间隔
    inter_arrivals = [2.0] * num_requests  # 固定2秒间隔
    
    # 运行实验
    cluster.run()
    env.process(put_requests_with_interarrivals(
        env, 
        cluster.scheduler, 
        inter_arrivals, 
        requests
    ))
    env.run(until=simulation_time)
    
    # 收集结果
    stats = cluster.get_cluster_stats()
    return stats

def run_throughput_comparison():
    """运行完整的吞吐量比较实验"""
    model_sizes = ["13b", "66b", "175b"]
    generation_lengths = [64, 128, 256, 512, 1024]
    systems = [
        ("distserve", None),      # Distserve_local
        ("distserve", "cxl"),     # CxlDistInfer
        ("vllm", None),           # vLLM_local
    ]
    
    results = {}
    for model in model_sizes:
        results[model] = {}
        for sys_name, offload_type in systems:
            sys_key = f"{sys_name}_{offload_type if offload_type else 'local'}"
            results[model][sys_key] = []
            
            for length in generation_lengths:
                stats = run_single_experiment(
                    model_size=model,
                    generation_length=length,
                    engine_type=sys_name,
                    offload_type=offload_type
                )
                
                # 计算平均吞吐量
                total_tokens = sum(stats["total_tokens"].values())
                total_time = max(worker.env.now for worker in cluster.get_all_workers())
                throughput = total_tokens / total_time
                results[model][sys_key].append(throughput)
                
                print(f"Model: {model}, System: {sys_key}, Length: {length}, Throughput: {throughput:.2f}")
    
    return results, generation_lengths

def plot_results(results: Dict, generation_lengths: List[int]):
    """绘制实验结果"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for idx, model in enumerate(results.keys()):
        ax = axes[idx]
        for sys_name, throughputs in results[model].items():
            ax.plot(generation_lengths, throughputs, marker='o', label=sys_name)
        
        ax.set_xlabel('Generation Length')
        ax.set_ylabel('Throughput (tokens/s)')
        ax.set_title(f'Model Size: {model}')
        ax.grid(True)
        ax.legend()
    
    plt.tight_layout()
    plt.savefig('throughput_comparison.png')
    plt.close()

if __name__ == "__main__":
    # 运行实验
    results, generation_lengths = run_throughput_comparison()
    
    # 绘制结果
    plot_results(results, generation_lengths)