from typing import TypedDict
import numpy as np
import pandas as pd

class Request_t(TypedDict):
    """请求基本信息类型"""
    req_id: int          # 请求ID
    prefill_lens: int    # 预填充长度
    output_lens: int     # 输出长度
    offload_type: str  # 新增：标识卸载类型##############


class RequestLog_t(TypedDict):
    """请求日志类型"""
    start_time: float    # 开始时间
    end_time: float      # 结束时间
    event_type: str      # 事件类型
    req_id: int         # 请求ID
    duration: float     # 持续时间
    location: str  # 新增：标识存储位置##############

class WorkerLog_t(TypedDict):
    """Worker日志类型"""
    start_time: float   # 开始时间
    end_time: float     # 结束时间
    event_type: str     # 事件类型
    worker_id: int      # Worker ID
    duration: float     # 持续时间
    decode_batch_size: int      # 解码批次大小
    prefill_batch: 'list[int]'  # 预填充批次信息
    decode_batch: 'list[int]'   # 解码批次信息
    offload_count: int  # 新增：卸载次数###################
    load_count: int     # 新增：加载次数##################

class LatencyDist_t(TypedDict):
    """延迟分布类型"""
    first_token_latency: float  # 首个token延迟
    decoding_latency: float     # 解码延迟
    tpot: float                # 每token平均时间(time per output token)
    inv_tpot_ms: float         # 每毫秒token速率
    inv_tpot_s: float          # 每秒token速率
    offload_latency: float    # 新增：卸载延迟#####################
    load_latency: float       # 新增：加载延迟####################
############################################################################
def calculate_per_request_latency(
    df: 'DataFrame[RequestLog_t]',
    output_lens: 'pd.Series' = None
) -> 'DataFrame[LatencyDist_t]':
    """计算每个请求的延迟统计"""
    # 获取关键时间点
    first_event = df[df.event_type == 'init'].groupby('req_id').start_time.min()
    first_wait_decode = df[df.event_type == 'wait_decode'].groupby('req_id').start_time.min()
    last_event = df[df.event_type == 'exit_system'].groupby('req_id').end_time.max()
    # 计算卸载和加载延迟
    offload_start = df[df.event_type == 'start_offload'].groupby('req_id').start_time
    offload_end = df[df.event_type == 'finish_offload'].groupby('req_id').end_time
    load_start = df[df.event_type == 'start_load'].groupby('req_id').start_time
    load_end = df[df.event_type == 'finish_load'].groupby('req_id').end_time

    # 基本各类延迟
    first_token_latency = first_wait_decode - first_event  # 第一个token的延迟
    decoding_latency = last_event - first_wait_decode      # 解码总延迟
    total_latency = last_event - first_event               # 总延迟
    
    
    # 创建结果DataFrame
    dist_df = pd.DataFrame({
        'first_token_latency': first_token_latency,
        'decoding_latency': decoding_latency,
        'total_latency': total_latency,
        'offload_latency': (offload_end - offload_start).fillna(0),
        'load_latency': (load_end - load_start).fillna(0)
    })


    # 如果提供了输出长度信息，计算吞吐量相关指标
    if output_lens is not None:
        # 计算每个输出token的平均时间
        tpot = decoding_latency.div(output_lens).replace([np.inf, - np.inf], 0)
        dist_df['tpot'] = tpot
        # 计算token生成速率（每毫秒/每秒）
        dist_df['inv_tpot_ms'] = 1 / tpot
        dist_df['inv_tpot_s'] = 1000 / tpot
    return dist_df
        
def organize_request_df(requests) -> 'DataFrame[Request_t]':
    """
    组织请求的基本信息到DataFrame
    
    将每个请求的基本属性(ID、预填充长度、输出长度)整理成表格形式
    """
    request_df = pd.DataFrame([
        {
            'req_id': r.req_id,
            'prefill_lens': r.prefill_lens,
            'output_lens': r.output_lens,
            'offload_type': r.offload_type  # 新增##############################
        }
        for r in requests
    ])
    return request_df

def transform_request_log_to_df(req: 'Request') -> 'DataFrame[RequestLog_t]':
    """
    将单个请求的日志转换为DataFrame格式
    
    参数:
    req: 请求对象
    
    返回:
    DataFrame包含：req_id, start_time, end_time, duration, event_type等信息
    """
    # 创建基本的DataFrame，包含时间、事件类型和worker_id
    df = pd.DataFrame(req.log, columns=['start_time', 'event_type', 'worker_id'])
    # 添加请求ID
    df['req_id'] = req.req_id
    # 计算每个事件的持续时间（当前事件到下一个事件的时间差）
    df['duration'] = df['start_time'].shift(-1) - df['start_time']
    df['duration'] = df['duration'].fillna(0)  # 最后一个事件持续时间设为0
    # 计算结束时间
    df['end_time'] = df['start_time'] + df['duration']
    # 添加位置信息#########################################
    location_events = df['event_type'].isin([
        'start_offload', 'finish_offload', 
        'start_load', 'finish_load'
    ])
    df.loc[location_events, 'location'] = req.location

    return df

def organize_request_event_df(requests) -> 'DataFrame[RequestLog_t]':
    """
    整合所有请求的事件日志到一个DataFrame
    
    将所有请求的日志合并成一个大表，方便统一分析
    """
    request_event_df = pd.concat([
        transform_request_log_to_df(r)
        for r in requests
    ])
    return request_event_df

def transform_worker_log_to_df(worker: 'Worker') -> 'DataFrame[WorkerLog_t]':
    """
    将单个worker的日志转换为DataFrame格式
    """
    if not worker.log:
        return None
        
    # 创建包含所有worker事件信息的DataFrame
    df = pd.DataFrame(worker.log, columns=[
        'start_time',      # 开始时间
        'event_type',      # 事件类型
        'num_tokens',      # token数量
        'prefill_bs',      # 预填充批大小
        'decode_bs',       # 解码批大小
        'prefill_batch',   # 预填充批次详情
        'decode_batch'     # 解码批次详情
    ])
    
    # 添加worker ID
    df['worker_id'] = worker.wid
    # 计算事件持续时间
    df['duration'] = df['start_time'].shift(-1) - df['start_time']
    df['duration'] = df['duration'].fillna(0)
    # 计算结束时间
    df['end_time'] = df['start_time'] + df['duration']
    return df

def organize_worker_event_df(cluster) -> 'DataFrame[WorkerLog_t]':
    """整合所有worker的事件日志"""
    worker_event_dfs = []
    for worker in cluster.get_all_workers():
        df = transform_worker_log_to_df(worker)
        if df is not None:
            # 添加卸载和加载统计
            offload_events = df['event_type'].str.contains('offload', na=False)
            load_events = df['event_type'].str.contains('load', na=False)
            df['offload_count'] = offload_events.cumsum()
            df['load_count'] = load_events.cumsum()
            worker_event_dfs.append(df)

    return pd.concat(worker_event_dfs) if worker_event_dfs else pd.DataFrame()