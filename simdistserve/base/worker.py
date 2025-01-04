"""
Worker class for simulation. One worker class manages a TP group.
"""
import random
import warnings
from collections import deque
from typing import Optional, List, Iterable, TYPE_CHECKING, Union, TypedDict, Literal
from uuid import UUID

from simdistserve.estimators.time_estimator import get_prefill_time, get_decode_time

if TYPE_CHECKING:
    from simdistserve.base.scheduler import Scheduler
    from simdistserve.base.request import Request


# TODO: (Refactor) Make this a configuration.
class WorkerConfig(TypedDict):
    """Behaviors of worker."""
    TP_Prefill: int  # Tensor parallelism for prefill (default = 1)
    TP_Decode: int  # Tensor parallelism for decode (default = 1)
    model_type: str  # Model type for prefill/decode time calculation (default = ModelType.opt_13b)
    prefill_max_batch_size: int  # Maximum number of prefill request in a batch (default = 10**7)
    decode_max_batch_size: int  # Maximum number of decode request in a batch (default = 10**7)
    prefill_max_tokens: int  # Max tokens in prefill iteration (default = 10**7)
    decode_max_tokens: int  # Max tokens in a iteration forward (default = 10**7)
    enable_chunked_prefill: Optional[bool]  # Enable memory pressure simulation (default = False)
    engine_type: Literal["distserve", "vllm"]  # Engine type for prefill/decode time calculation (default = "distserve")

    # TODO: Deprecated
    TP: Optional[int]  # Tensor parallelism (default = 1)
    pass


class Worker:
    def __init__(
        self, env, wid,
        cluster=None,
        is_last_in_pipeline: bool = False,
        pipe_rank: int = None,
        should_request_stay: bool = True,
        prefill_max_batch_size: int = 10 ** 7,
        decode_max_batch_size: int = 10 ** 7,
        global_scheduler: 'Scheduler' = None,
        model_type: str = None,
        TP: int = 1,
        TP_Prefill: int = None,
        TP_Decode: int = None,
        enable_chunked_prefill=False,
        prefill_max_tokens=10 ** 7,
        decode_max_tokens=10 ** 7,
        decode_back_pressure: float = 0.9,
        engine_type: Literal["distserve", "vllm"] = "distserve",
        #########################    #########################
        memory_threshold: float = 0.8,  
        cxl_load_time_per_mb: float = 0.0078125,  
        local_load_time_per_mb: float = 0.03125,
        offload_type: str = 'cxl',  # 'cxl', 'local', None
        gpu_memory_size = 32 * 1024, ##32GB
        cxl_memory_size = 700 * 1024, ##700GB
        local_memory_size = 64 * 1024, ##64GB
    ):
        self.env = env
        self.cluster = cluster  # Refer to the cluster of init.
        self.wid = wid
        self.pipe_rank = pipe_rank
        self.is_last_in_pipeline = is_last_in_pipeline
        self.next_worker: 'Optional[Worker]' = None
        self.model_type = model_type

        # TODO: (Deprecate) TP should be deprecate in favor of TP_prefill and TP_decode.
        self.TP = TP
        self.TP_Prefill = TP_Prefill
        self.TP_Decode = TP_Decode
        if (self.TP_Prefill is None) and (self.TP_Decode is None):
            warnings.warn(f"TP_Prefill and TP_Decode are not set. Default to {TP = } only apply to prefill.")
            self.TP_Prefill = TP
            self.TP_Decode = 1
        elif (self.TP_Prefill is not None) and (self.TP_Decode is not None):
            # Using the new TP_prefill and TP_decode value, instead of TP.
            pass
        elif (self.TP_Prefill is None) or (self.TP_Decode is None):
            warnings.warn(f"{TP = } will be deprecated soon. Use TP_Prefill and TP_Decode.")
            self.TP_Prefill = TP
            self.TP_Decode = 1
            pass

        # Same request should stay in the same worker.
        # If set to false, then it will forward to the global scheduler.
        self.global_scheduler = global_scheduler
        self.should_request_stay: bool = should_request_stay
        # Maximum number requests to fill in prefill batch. (Default 0 => 10 ** 7, big enough number)
        self.prefill_max_batch_size: int = prefill_max_batch_size if prefill_max_batch_size > 0 else 10 ** 7
        self.decode_max_batch_size: int = decode_max_batch_size if decode_max_batch_size > 0 else 10 ** 7
        # Maximum number of tokens for a prefill request to batch.
        self.prefill_max_tokens: int = prefill_max_tokens if prefill_max_tokens > 0 else 10 ** 7
        self.decode_max_tokens: int = decode_max_tokens if decode_max_tokens > 0 else 10 ** 7
        # Enable chunked prefill (if True) or prioritization scheduling (if False)
        self.enable_chunked_prefill: bool = enable_chunked_prefill
        # Decode worker stop accepting incoming request when this is full.
        self.decode_back_pressure = decode_back_pressure

        self.prefill_queue: 'deque[Request]' = deque()
        self.decode_queue: 'deque[Request]' = deque()
        self._prefill_ips: int = 0  # Elements in progress for prefill
        self._decode_ips: int = 0  # Elements in progress for decode
        self._wakeup_event = env.event()
        self.log: 'list[tuple[float, str, int, int, int, list[int], list[int]]]' = []

        # Simulate scheduler delay in terms of number of decode rounds.
        self._prefill_sched_delay: int = 0
        self.engine_type = engine_type
        
        self.offload_type = offload_type
        self.memory_threshold = memory_threshold
        self.load_memory_threshold = 0.8
        self.cxl_load_time_per_mb = cxl_load_time_per_mb
        self.local_load_time_per_mb = local_load_time_per_mb
        self.max_tokens_limit = self.decode_max_tokens
        self.gpu_memory_size = gpu_memory_size
        self.cxl_memory_size = cxl_memory_size
        self.local_memory_size = local_memory_size
        self.cxl_memory_used = 0
        self.local_memory_used = 0
        self.TPOP = 0
        self.reset_stats()  
        pass

    def reset_stats(self):
        
        self.stats = {
            'total_delay': 0,
            'total_tokens': 0,
            'offload_amount': 0,
            'load_amount': 0,
            'max_gpu_usage': 0,
            'max_gpu_memory_usage': 0,  
            'request_count': 0,      
            'offload_count': 0,     
            'load_count': 0,       
            'avg_batch_size': 0  
        }

    @property
    def is_first_in_pipeline(self):
        return self.pipe_rank == 0

    @property
    def has_back_pressure(self) -> bool:
        threshold = int(self.decode_max_batch_size * self.decode_back_pressure)
        return sum(r.current_context_len for r in self.decode_queue) > threshold

    def __repr__(self):
        return f"Worker {self.wid}"

    def _log_event(self, event, num_tokens: int = 0, prefill_bs=0, decode_bs=0,
                   prefill_len_list=None, decode_len_list=None):
        if prefill_len_list is None:
            prefill_len_list = []
        if decode_len_list is None:
            decode_len_list = []
        item = (self.env.now, event, num_tokens, prefill_bs, decode_bs, prefill_len_list, decode_len_list)
        self.log.append(item)
        # print(item)
        return


    def run(self):
        print(f"Worker {self.wid} starting...")
        while True:
            if not (self.prefill_queue or self.decode_queue):
                print(f"Worker {self.wid} waiting for requests...")
                yield self._wakeup_event

            if self.prefill_queue and not self.has_back_pressure:
                print(f"Worker {self.wid} doing prefill...")
                yield from self.do_prefill()
            else:
                print(f"Worker {self.wid} doing decode...")
                yield from self.do_decode()

            self._log_event("wait")
            pass

        pass

    def run_offload(self):
        while True:
            #yield self._wakeup_event
            if self.decode_queue and self.check_memory_pressure:
                print(f"Worker {self.wid} doing offload")
                yield from self.select_requests_to_offload()
            else:
                pass
            pass
        pass

    
    def run_load(self):
        while True:
            #yield self._wakeup_event
            if self.decode_queue and not self.check_memory_pressure:
                print(f"Worker {self.wid} doing load")
                yield from self.select_requests_to_load()
            else:
                pass
            pass
        pass

    def add_ray_overhead(self, sum_of_tokens) -> int:
        base_overhead = 2
        k = 0.0001
        delay = base_overhead + sum_of_tokens * k
        return delay

    # run = run_with_schedule_delay

    def wakeup(self):
        self._wakeup_event.succeed()
        self._wakeup_event = self.env.event()
        return
    
    ######################### Modified #########################

    def gpu_memory_usage(self):
        if self.engine_type == "vllm":
            return sum(r.current_kvcache_size for r in (self.prefill_queue + self.decode_queue) if r.location == "gpu")/self.gpu_memory_size
        else:
            return sum(r.current_kvcache_size for r in self.decode_queue if r.location == "gpu")/self.gpu_memory_size


    def check_memory_pressure(self) -> bool:
        
        current_gpu_usage = self.gpu_memory_usage()
        print(f"Current GPU memory usage: {current_gpu_usage:.4f}")
        return current_gpu_usage > self.memory_threshold

    # def select_requests_to_offload(self):
    #     """选择需要卸载的请求"""
    #     if self.offload_type == 'cxl':
    #         # CXL方案：基于优先级的选择
    #         sorted_requests = sorted(self.decode_queue, key=lambda req: req._calculate_priority()) #升序
    #     else:  # local或无卸载
    #         # 本地内存方案：简单FIFO
    #         sorted_requests = list(self.decode_queue)

    #     to_offload = []
    #     current_tokens = sum(req.current_context_len for req in self.decode_queue)
        
    #     # 卸载直到内存使用低于阈值
    #     for req in sorted_requests:
    #         if current_tokens <= (self.max_tokens_limit * self.memory_threshold):
    #             break
    #         to_offload.append(req)
    #         current_tokens -= req.current_context_len
            
    #     return to_offload

    # def select_requests_to_load(self,requests_to_offload):
    #     """ select request to load back to GPU """
    #     to_load = []
    #     if self.offload_type == 'cxl':
    #         # CXL load type
    #         sorted_requests = sorted(requests_to_offload, key=lambda req: req._Calculate_Priority(), reverse=True) #降序
    #     else:
    #         # local load type
    #         sorted_requests = list(requests_to_offload)
        
    #     for req in sorted_requests:
    #         if self.gpu_memory_usage() * self.gpu_memory_size + req.current_kv_cache_size > self.memory_threshold * self.gpu_memory_size:
    #             break
    #         to_load.append(req)
    #     return to_load

    def select_requests_to_offload(self):
        
        if not self.offload_type:  
            return []
            
        to_offload = []
        if self.offload_type == 'cxl':
            
            if not self.check_memory_pressure():
                return []
                
            sorted_requests = sorted(
                [req for req in self.decode_queue if req.location == 'gpu'],
                key=lambda req: req._calculate_priority()
            )  

            for req in sorted_requests:
                if self.check_memory_pressure():
                    req_size = req.current_kvcache_size
                    if self.check_cxl_memory_available(req_size):
                        to_offload.append((req, 'cxl'))
                        self.update_memory_usage(req, req.location, 'cxl')
                        self.stats['offload_amount'] += req.current_kvcache_size
                        self.stats['offload_count'] += 1
                        self.cxl_memory_used += req_size
                    else:
                        break  
                else:
                    break 
            self.stats['max_gpu_memory_usage'] = max(self.stats['max_gpu_memory_usage'], self.gpu_memory_usage())
        elif self.offload_type == 'local':
            
            if not self.check_memory_pressure():
                return []

            for req in self.decode_queue:
                if req.location != 'gpu':
                    continue
                if self.check_memory_pressure():    
                    req_size = req.current_kvcache_size
                    if self.check_local_memory_available(req_size):
                        to_offload.append((req, 'local'))
                        self.update_memory_usage(req, req.location, 'local')
                        self.stats['offload_amount'] += req.current_kvcache_size
                        self.stats['offload_count'] += 1
                        self.local_memory_used += req_size
                    else:
                        break  
                else:
                    break  
            self.stats['max_gpu_memory_usage'] = max(self.stats['max_gpu_memory_usage'], self.gpu_memory_usage())

        yield self.env.timeout(0)
        return to_offload

    def select_requests_to_load(self):
        
        if not self.offload_type: 
            return []
            
        to_load = []
        

        if self.offload_type == 'cxl':
           
            offloaded_reqs = [req for req in self.decode_queue if req.location == 'cxl']
            sorted_reqs = sorted(
                offloaded_reqs,
                key=lambda req: req._calculate_priority(),
                reverse=True  
            )

            for req in sorted_reqs:
                if self.gpu_memory_usage() + req.current_kvcache_size/self.gpu_memory_size > self.load_memory_threshold:
                    break
                to_load.append(req)
                self.update_memory_usage(req, req.location, 'gpu')
                delay = req.current_kvcache_size * self.cxl_load_time_per_mb
                self.stats['total_delay'] += delay
                self.stats['load_amount'] += req.current_kvcache_size
                self.stats['load_count'] += 1
                self.cxl_memory_used -= req.current_kvcache_size
                #current_gpu_usage += req.current_kvcache_size/self.gpu_memory_size
                yield self.env.timeout(delay)
            self.stats['max_gpu_memory_usage'] = max(self.stats['max_gpu_memory_usage'], self.gpu_memory_usage())

        elif self.offload_type == 'local':
            
            offloaded_reqs = [req for req in self.decode_queue if req.location == 'local']
            for req in offloaded_reqs: 
                if self.gpu_memory_usage() + req.current_kvcache_size/self.gpu_memory_size > self.load_memory_threshold:
                    break
                to_load.append(req)
                self.update_memory_usage(req, req.location, 'gpu')
                delay = req.current_kvcache_size * self.local_load_time_per_mb
                self.stats['total_delay'] += delay
                self.stats['load_amount'] += req.current_kvcache_size
                self.stats['load_count'] += 1
                self.local_memory_used -= req.current_kvcache_size
                #current_gpu_usage += req.current_kvcache_size/self.gpu_memory_size
                yield self.env.timeout(delay)
            self.stats['max_gpu_memory_usage'] = max(self.stats['max_gpu_memory_usage'], self.gpu_memory_usage())

        return to_load


    
#    def calculate_load_delay(self, requests):
#        """计算加载延迟
#        根据不同的存储类型计算不同的加载延迟
#        """
#        if not self.offload_type:#or not requests:  # 无卸载方案
#            return 0
#            
#        total_delay = 0
#        print(f"\n[Worker {self.wid}] Calculating load delay:")
#        print(f"- Number of requests to load: {len(requests)}")
#        for req in requests:
#            req_size_mb = req.current_kvcache_size
#            #print(f"LOAD REQUEST POSITION:{req.location},REQ_SIZE:{req_size_mb}")
#            if req.location == 'cxl':
#                # CXL加载延迟
#                req_delay = req_size_mb * self.cxl_load_time_per_mb  # 使用MB单位计算延迟
#                print(f"  CXL load - Size: {req_size_mb:.2f}MB, Delay: {req_delay:.2f}s")
#                total_delay += req_delay
#                # total_delay += (req.current_kvcache_size * self.cxl_load_time_per_mb)
#            elif req.location == 'local':
#                # 本地内存加载延迟
#                
#                req_delay = req_size_mb * self.local_load_time_per_mb  # 使用MB单位计算延迟
#                print(f"  Local load - Size: {req_size_mb:.2f}MB, Delay: {req_delay:.2f}s")
#                total_delay += req_delay
#                # total_delay += (req.current_kvcache_size * self.local_load_time_per_mb)
#                    
#        print(f"Total load delay: {total_delay:.2f}s")
#        return total_delay
    

    def check_gpu_memory_available(self, request_size: int) -> bool:
        
        return (self.gpu_memory_usage() + request_size/self.gpu_memory_size) <= self.memory_threshold

    def check_cxl_memory_available(self, request_size: int) -> bool:
       
        return (self.cxl_memory_used + request_size) <= self.cxl_memory_size

    def check_local_memory_available(self, request_size: int) -> bool:
      
        return (self.local_memory_used + request_size) <= self.local_memory_size

    def update_memory_usage(self, request, old_location: str, new_location: str):

        size = request.current_kvcache_size

        if old_location == 'cxl':
            self.cxl_memory_used -= size
        elif old_location == 'local':
            self.local_memory_used -= size
            
        
        if new_location == 'cxl':
            self.cxl_memory_used += size
        elif new_location == 'local':
            self.local_memory_used += size

        
        request.move_to_storage(new_location)



    #######################  End Modified ###########################

    def forward_prefill(self, items):
        # if items is not iterable, then make it iterable
        if not items:
            return
        if not isinstance(items, Iterable):
            items = [items]

        self.next_worker.prefill_queue.extend(items)
        self.next_worker.wakeup()
        return

    def forward_decode(self, items: Union['Request', Iterable['Request']], to_scheduler: bool = False):
        if not items:
            return
        if not isinstance(items, Iterable):
            items = [items]

        if not to_scheduler:
            self.next_worker.decode_queue.extend(items)
            self.next_worker.wakeup()
            return

        for item in items:
            self.global_scheduler.schedule_decode(item)
        return

    def _enter_decodes(self, remaining_tok_in_batch: int) -> 'List[Request]':
        # decode_max_tokens

        # Acceptable decode requests is capped by the remaining allowed tokens in this batch.
        # TODO: Hack: Must revert this to use the max token given
        # watermark = 0.9
        # decode_max_tokens = self.decode_max_tokens * watermark
        decode_max_tokens = 65536
        decode_reqs = []
    
        print(f"Starting _enter_decodes with queue length: {len(self.decode_queue)}")
    
        try:
            queue_list = list(self.decode_queue)
            to_remove = []  
    
            for idx, req in enumerate(queue_list):
                if req.location == "gpu":
                    if (req.current_context_len + 1) > decode_max_tokens:
                        print(f"Reached token limit. Current: {req.current_context_len + 1}, Max: {decode_max_tokens}")
                        break
                
                    decode_max_tokens -= (req.current_context_len + 1)
                    decode_reqs.append(req)
                    to_remove.append(idx)
            self.decode_queue = deque([req for i, req in enumerate(queue_list) if i not in to_remove])
        except Exception as e:
            print(f"Error in _enter_decodes: {e}")
            return []  
    
        print(f"Finished _enter_decodes, selected {len(decode_reqs)} requests")
        return decode_reqs


        # while idx < _decode_len:
        #     req = self.decode_queue[idx]
        #     if  req.location == "gpu":
        #         if (req.current_context_len + 1) > decode_max_tokens:
        #             break
        #         decode_max_tokens -= (req.current_context_len + 1)
        #         decode_reqs.append(self.decode_queue[idx])
        #         del self.decode_queue[idx]
        #         idx -= 1
        #     idx += 1
        #     if len(self.decode_queue) <= 0:
        #         break
        
        # for r in decode_reqs:
        #     r.do_decode(wid=self.wid)
        
        #return decode_reqs


    def _enter_prefill(self) -> 'List[Request]':
        result: 'List[Request]' = []

        # Limit the maximum prefill requests to handle.
        max_request_size = min(self.prefill_max_batch_size, len(self.prefill_queue))

        # TODO: (Refactor) This logic becomes spaghetti.
        # If worker is not the first in pipeline, then it will just identify the chunks of prefill.
        if not self.is_first_in_pipeline:
            # Then just fetch all decode with the same chunk-id.
            chunk_id = self.prefill_queue[0].chunk_id
            for i in range(max_request_size):
                candidate: 'Request' = self.prefill_queue[0]
                if candidate.chunk_id != chunk_id:
                    break
                result.append(self.prefill_queue.popleft())
            pass

        else:
            # Worker is the first in pipeline, then it will do chunked prefill.
            chunk_size = 0
            prefill_max_tokens = self.prefill_max_tokens
            # chunk_id assign as uuid
            chunk_id = UUID(int=random.getrandbits(128))
            for _ in range(max_request_size):
                candidate: 'Request' = self.prefill_queue[0]

                if self.enable_chunked_prefill:
                    # The prefill portion that we picked from the candidate.
                    sched_size = min(
                        # The to-schedule size is the minimum of
                        # (1) the remaining prefill size of the candidate, and
                        # (2) the maximum allowed size of a chunked-prefill batch.
                        # This way we greedily cut and schedule the prefill chunk.
                        candidate.remain_prefill_lens,
                        prefill_max_tokens - chunk_size  # max batch size in a chunked-prefill batch - chunk size
                    )
                    if sched_size <= 0:
                        break
                else:
                    # If the whole request can fit into the chunk,
                    # then just schedule the whole request.
                    sched_size = candidate.remain_prefill_lens
                    if sched_size > prefill_max_tokens:
                        break
                    pass

                # Candidate is picked. Now fill in the chunked-prefill information.
                candidate.current_prefill_lens = sched_size
                candidate.remain_prefill_lens -= sched_size
                prefill_max_tokens -= sched_size
                candidate.chunk_id = chunk_id
                chunk_size += sched_size
                assert candidate.remain_prefill_lens >= 0
                result.append(self.prefill_queue.popleft())
                pass
        for i in result:
            i.do_prefill(wid=self.wid)
        return result

    def _exit_prefill(self, prefill_items: List['Request']):
        for item in prefill_items:
            next_wid = self.next_worker.wid if self.next_worker else None
            item.finish_prefill(is_finished_one_round=self.is_last_in_pipeline, wid=self.wid, next_wid=next_wid)
            if not self.is_last_in_pipeline or (item.remain_prefill_lens > 0):
                # Finish one chunk of prefill. Now forward to the next worker
                # (or head of worker) to do the rest of the parts.
                self.forward_prefill(item)
                continue

            # Arrive at worker who is at the last of pipeline.
            if item.should_finish():
                # ... just a sanity check to avoid any infinite loop.
                continue
            self.forward_decode(item, to_scheduler=(not self.should_request_stay))
        return

    def _exit_decode(self, decode_reqs):
        if not decode_reqs:
            print(f"Worker {self.wid}: No decode requests to process")
            return
        next_wid = self.next_worker.wid if self.next_worker else None
        print(f"Worker {self.wid}: Processing {len(decode_reqs)} decode requests")
        for r in decode_reqs:
            r.finish_decode(is_finished_one_round=self.is_last_in_pipeline, next_wid=next_wid)
            
        next_decode_batch = [r for r in decode_reqs if not r.should_finish()]
        if next_decode_batch:
            print(f"Worker {self.wid}: Forwarding {len(next_decode_batch)} requests to scheduler")
            self.forward_decode(next_decode_batch, to_scheduler=True)  
        else:
            print(f"Worker {self.wid}: All requests completed, nothing to forward")

        return
        # next_decode_batch = tuple(r for r in decode_reqs if not r.should_finish())
        # self.forward_decode(next_decode_batch)
        # return

    def do_prefill(self):
        prefill_items: 'List[Request]' = self._enter_prefill()
        if self.enable_chunked_prefill:
            remaining_tok_in_batch = self.prefill_max_tokens - sum(x.current_prefill_lens for x in prefill_items)
            decode_reqs = self._enter_decodes(remaining_tok_in_batch)
        else:
            decode_reqs = []
        # TODO: (Refactor) The `num_tokens` may be used inaccurately in the get prefill time function.
        num_tokens = sum(x.current_prefill_lens for x in prefill_items)
        num_tokens += len(decode_reqs)

        self._log_event(
            "do_prefill",
            num_tokens=num_tokens,
            prefill_bs=len(prefill_items),
            decode_bs=len(decode_reqs),
            prefill_len_list=[x.current_prefill_lens for x in prefill_items],
            decode_len_list=[x.current_context_len for x in decode_reqs],
        )

        # Get prefill time wrt total number of tokens.
        delay = get_prefill_time(
            num_tokens,
            bs=len(prefill_items),
            decode_bs=len(decode_reqs),
            pp=self.cluster.PP_prefill,
            model_type=self.model_type, TP=self.TP_Prefill,
            prefill_len_list=[x.current_prefill_lens for x in prefill_items],
            engine_type=self.engine_type,
            # __prefill_reqs=prefill_items,
            # __decode_reqs=decode_reqs,
        )
        num_tokens = sum(x.current_context_len for x in (prefill_items + decode_reqs))
        if self.is_first_in_pipeline:
            delay += self.add_ray_overhead(num_tokens)
        # Set the number of prefills in progress such that the scheduler get proper information about the worker.
        self._prefill_ips = len(prefill_items)
        yield self.env.timeout(delay)
        self._prefill_ips = 0
        self._exit_prefill(prefill_items)
        self._exit_decode(decode_reqs)
        return

    def do_decode(self):
        
        # 1.
        print(f"Worker {self.wid} starting decode process...")
        #self.stats['max_gpu_memory_usage'] = max(self.stats['max_gpu_memory_usage'], self.gpu_memory_usage())
        print(f"Worker {self.wid} checking memory pressure...")

        #requests_to_offload = []  
        #requests_to_load = []      
        

        # 2.
        #if self.offload_type and self.check_memory_pressure():
        #    print(f"Memory pressure detected, proceeding with offload...")
        #    requests_to_offload = self.select_requests_to_offload()
        #    self.stats['offload_count'] += len(requests_to_offload)
           


        # 3. 
        
        decode_reqs = self._enter_decodes(self.decode_max_tokens)
        #except Exception as e:
        #    print(f"Error in getting decode requests: {e}")
        #    decode_reqs = []  
        

        batch_size = len(decode_reqs)
        self.stats['request_count'] += batch_size
        self.stats['avg_batch_size'] = (
            self.stats['total_tokens'] / self.stats['request_count'] 
            if self.stats['request_count'] > 0 else 0
    )

        
        print(f"Processing batch of {batch_size} requests")
        
        self._log_event(
            "do_decode", 
            num_tokens=batch_size, 
            decode_bs=batch_size,
            decode_len_list=[x.current_context_len for x in decode_reqs],
        )
        
        _token_generated_list = [x.current_context_len + 1 for x in decode_reqs]
        print(f"Total tokens to generate: {sum(_token_generated_list)}")
        
        delay = get_decode_time(
            batch_size, 
            pp=self.cluster.PP_decode,
            model_type=self.model_type, 
            TP=self.TP_Decode,
            token_generated_list=_token_generated_list,
            engine_type=self.engine_type,
        )
        print(f"Base decode delay: {delay:.2f}s")
        #self.stats['total_delay'] = delay
        
        # 4. 
        #if self.offload_type:
        #
        #    requests_to_load = self.select_requests_to_load() #modified:changed 'decode_reqs'cause all these requests have label 'gpu' due to '_enter'
        #    self.stats['load_count'] += len(requests_to_load)
        #    print(f"XXXXX{len(requests_to_load)}")
        
           
    
        # 5.
        num_tokens = sum(x.current_context_len for x in decode_reqs)
        self.stats['total_tokens'] += num_tokens
        if self.is_first_in_pipeline:
            delay += self.add_ray_overhead(num_tokens)
        self.stats['total_delay'] += delay
        print(f"Total delay for this cycle: {self.stats['total_delay']:.2f}s")
        self.TPOP = delay / num_tokens if num_tokens > 0 else 0

        # 6. 
        yield self.env.timeout(delay)
        self._exit_decode(decode_reqs)

        return




    pass




###################################
