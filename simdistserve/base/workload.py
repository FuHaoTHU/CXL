import marshal
import random
from contextlib import contextmanager
from os import PathLike
from typing import List

import numpy as np

from simdistserve.base.request import Request


@contextmanager
def numpy_seed(seed):
    if seed is None:
        yield
        return

    state = np.random.get_state()  # Save the current state
    try:
        np.random.seed(seed)  # Set the new seed
        yield
    finally:
        np.random.set_state(state)  # Restore the original state


def convert_interarrival_to_absolutearrival(x: 'List[float]'):
    y = 0
    result = []
    for t in x:
        y += t
        result.append(y)
    return result


def convert_absolutearrival_to_interarrival(x: 'List[float]'):
    result = [0]
    for i in range(1, len(x)):
        delay = x[i] - x[i - 1]
        delay *= 1000
        result.append(delay)
    return result


def convert_pd_pair_to_request(pairs: 'list[tuple[int, int]]') -> 'List[Request]':
    result = []
    for i, (prefill_length, output_length) in enumerate(pairs):
        r = Request(
            env=None,
            req_id=i,
            prefill_length=prefill_length,
            output_lens=output_length,
        )
        result.append(r)
    return result


class NamedList(list):
    def __init__(self, *args):
        super().__init__(*args)
        self.name = None

    def set_name(self, name):
        self.name = name
        return self


def get_fixed_interarrival(n, delay: float):
    """Fixed interarrival delay (ms)."""
    assert n > 0
    data = [0] + [delay] * (n - 1)
    result = NamedList(data).set_name(f'fixed(delay={delay})')
    return result


def get_poisson_interarrival(n: int, rate: float, seed=None):
    """
    Return the list of inter-arrival time (ms).
    Note: the 0-th element is 0 - the first request always have 0-delay.
    See the processing function for why.
    """
    return get_gamma_interarrival(n, rate, 1, seed=seed)


def get_gamma_interarrival(n: int, rate: float, cv: float, seed=None):
    assert n > 0
    with numpy_seed(seed):
        shape = 1 / (cv * cv)
        scale = cv * cv / rate
        result = np.random.gamma(shape, scale, size=n - 1)
        result *= 1000

    data = [0] + list(result)
    result = NamedList(data).set_name(f'gamma(rate={rate}, cv={cv}, seed={seed})')
    return result
###########################################################################################################
def apply_long_tail_distribution(requests: List[Request], mean: float = 4, sigma: float = 1) -> List[Request]: ##doesn't make such a big difference
    """
    对请求的output_lens应用长尾分布处理
    使用对数正态分布：长的更长，短的更短
    
    参数:
    requests: 原始请求列表
    mean: 对数正态分布的均值参数
    sigma: 对数正态分布的标准差参数
    """
    # 对请求按output_lens排序
    sorted_requests = sorted(requests, key=lambda x: x.output_lens)
    
    # 生成长尾分布的新长度
    n = len(requests)
    new_lengths = np.random.lognormal(mean=mean, sigma=sigma, size=n)
    new_lengths = np.sort(new_lengths)  # 排序以保持相对顺序
    
    # 将新长度规范化到合理范围
    min_len = min(r.output_lens for r in requests)
    max_len = max(r.output_lens for r in requests)
    new_lengths = (new_lengths - new_lengths.min()) / (new_lengths.max() - new_lengths.min())
    new_lengths = new_lengths * (max_len - min_len) + min_len
    new_lengths = new_lengths.astype(int)
    
    # 更新请求的output_lens
    for req, new_len in zip(sorted_requests, new_lengths):
        req.output_lens = new_len 
    
    return sorted_requests

# 修改sample_requests函数
def sample_requests_1(dataset_path: PathLike, num_prompts: int) -> 'list[(int, int)]':
    """
    从数据集采样并应用长尾分布
    """
    with open(dataset_path, 'rb') as f:
        dataset = marshal.load(f)
    dataset = dataset['reqs']
    result = random.sample(dataset, num_prompts)
    result = [(p, d) for (_, p, d) in result]

    # 生成请求列表
    requests = [
        Request(
            env=None,
            req_id=i,
            prefill_length=prefill_length,
            output_lens=output_length,
        )
        for i, (prefill_length, output_length) in enumerate(result)
    ]
    
    # 应用长尾分布处理
    requests = apply_long_tail_distribution(requests)
    
    return requests
##################################################
def sample_requests(dataset_path: PathLike, num_prompts: int) -> 'list[(int, int)]':
    """
    sample_requests: Sample the given number of requests from the dataset.
    :param dataset_path: The path to the dataset.
    :param num_prompts: The number of prompts to sample.
    :return: A list of prompts and decode lengths.
    """
    with open(dataset_path, 'rb') as f:
        # {dataset_name:str, data:list[(prompt:str, prompt_len:int, output_len:int)]}
        dataset = marshal.load(f)
    dataset = dataset['reqs']
    result = random.sample(dataset, num_prompts)
    
    result = [(p, d) for (_, p, d) in result]

    #### test random dataset #################
    sum = 0
    for i, (prefill_length, output_length) in enumerate(result):
        sum += prefill_length + output_length
        print(f"TEST RANDOM request {i}: ({prefill_length},{output_length})")
    
    print(f"TEST RANDOM: {sum}")

    # Generate requests
    requests = [
        Request(
            env=None,
            req_id=i,
            prefill_length=prefill_length, #max:+1925
            output_lens=output_length,
        )
        for i, (prefill_length, output_length) in enumerate(result)
    ]
    return requests
