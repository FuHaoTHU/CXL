# Fit a model where prefill does not have an intercept, and decode does have one.
import json
from pathlib import Path

from simdistserve.constants import ModelTypes


def load_distserve_profile_data():
   profile_data_path = Path(__file__).parent / "profile_data" / "profiler-a100-80g.distserve.json"
   with open(profile_data_path) as f:
       profile_data = json.load(f)
       # 验证所需模型大小的数据
    #    required_models = ["opt_13b", "opt_30b"]
    #    for model in required_models:
    #        if model not in profile_data:
    #            raise ValueError(f"Missing profile data for {model}")
       return profile_data


def load_vllm_profile_data():
   profile_data_path = Path(__file__).parent / "profile_data" / "profiler-a100-80g.vllm.json"
   with open(profile_data_path) as f:
       profile_data = json.load(f)
       # 验证所需模型大小的数据
    #    required_models = ["opt_13b", "opt_30b"]
    #    for model in required_models:
    #        if model not in profile_data:
    #            raise ValueError(f"Missing profile data for {model}")
       return profile_data


distserve_profile_data = load_distserve_profile_data()
vllm_profile_data = load_vllm_profile_data()


def get_prefill_time(
   num_tokens=None, 
   pp=1, 
   bs=1, 
   decode_bs=0, 
   model_type=ModelTypes.opt_13b,
   TP=1,
   prefill_len_list=None,
   engine_type="distserve",
   offload_type=None,
   **kw
):
   """计算预填充阶段的时间
   
   Args:
       num_tokens: token总数
       pp: pipeline parallelism degree
       bs: batch size
       decode_bs: decode batch size  
       model_type: 模型类型
       TP: tensor parallelism degree
       prefill_len_list: prefill长度列表
       engine_type: 引擎类型 ("distserve" or "vllm")
       offload_type: 卸载类型 ("cxl", "local" or None)
   """
   # 获取基础性能参数
   if engine_type == "distserve":
       params = distserve_profile_data[ModelTypes.formalize_model_name(model_type)][str(TP)]
       a, b, c = params["prefill"]
   else:
       params = vllm_profile_data[ModelTypes.formalize_model_name(model_type)][str(TP)]
       a, b, c = params["prefill"]

   # 基础时间计算
   f = 1  # 缩放因子
   a, b, c = (a * f, b * f, c * f)
   pp_factor = 1 / pp
   pp_const = 1 * pp  # Pipeline parallelism开销
   
   num_total_tokens = sum(prefill_len_list)
   sum_num_tokens_sqr = sum([x ** 2 for x in prefill_len_list])
   
   delay = a + b * num_total_tokens + c * sum_num_tokens_sqr
   delay = delay * pp_factor + pp_const

   return delay


def get_decode_time(
   num_requests,
   pp=1,
   model_type=ModelTypes.opt_13b,
   TP=1, 
   token_generated_list=None,
   engine_type="distserve",
   offload_type=None,
   **kw
):
   """计算解码阶段的时间
   
   Args:
       num_requests: 请求数量
       pp: pipeline parallelism degree  
       model_type: 模型类型
       TP: tensor parallelism degree
       token_generated_list: 每个请求生成的token数列表
       engine_type: 引擎类型 ("distserve" or "vllm")
       offload_type: 卸载类型 ("cxl", "local" or None)
   """
   batch_size = num_requests

   # 获取基础性能参数
   if engine_type == "distserve":
       params = distserve_profile_data[ModelTypes.formalize_model_name(model_type)][str(TP)]
       threshold = params["decoding_large_small_bs_threshold"]
       if batch_size < threshold:
           a, b, c = params["decoding_smallbs"]
       else:
           a, b, c = params["decoding_largebs"]
   else:
       params = vllm_profile_data[ModelTypes.formalize_model_name(model_type)][str(TP)]
       threshold = params["decoding_large_small_bs_threshold"]
       if batch_size < threshold:
           a, b, c = params["decoding_smallbs"]
       else:
           a, b, c = params["decoding_largebs"]

   # 基础时间计算
   f = 1  # 缩放因子
   pp_factor = 1 / pp
   pp_const = 0  # Pipeline parallelism开销
   
   num_total_tokens = sum(token_generated_list)
   
   delay = a + b * num_total_tokens + c * batch_size
   delay = delay * pp_factor + pp_const
   delay *= f

   return delay