# """
# Scheduler class for simulation.
# Handles both prefill and decode scheduling with memory management support.
# """
# from queue import Queue
# from typing import List, TYPE_CHECKING, Tuple, Union

# if TYPE_CHECKING:
#     from simdistserve.base.request import Request
#     from simdistserve.base.worker import Worker

# class Scheduler:
#     def __init__(self, 
#                  env, 
#                  prefill_heads, 
#                  decode_heads,
#                  memory_threshold: float = 0.8):  # 添加内存阈值参数
#         print(f"\nInitializing Scheduler:")
#         print(f"Number of prefill heads: {len(prefill_heads)}")
#         print(f"Number of decode heads: {len(decode_heads)}")
#         print("Decode heads worker IDs:", [w.wid for w in decode_heads])
#         self.env = env
#         self._prefill_heads: 'List[Worker]' = prefill_heads
#         self._prefill_queues = [i.prefill_queue for i in self._prefill_heads]
#         self._decode_heads: 'List[Worker]' = decode_heads
#         self._decode_queues = [i.decode_queue for i in self._decode_heads]
#         self.memory_threshold = memory_threshold
#         pass

#     def _calculate_worker_load(self, worker: 'Worker', queue) -> float:
#         """计算worker的负载分数
#         考虑:
#         1. 队列长度
#         2. 正在处理的任务数
#         3. GPU内存使用情况
#         """
#         queue_load = len(queue)
#         processing_load = worker._prefill_ips
#         memory_load = worker.gpu_memory_usage()
        
#         # 当内存接近阈值时，显著增加负载分数
#         if memory_load > self.memory_threshold:
#             memory_penalty = 100  # 添加较大惩罚
#         else:
#             memory_penalty = 0
            
#         return queue_load + processing_load + memory_penalty

#     def _find_best_worker_and_queue(self, workers, queues) -> 'Tuple[Worker, Union[Queue, List]]':
#         """找到负载最小的worker"""
#         print("\nScheduler selecting worker:")  # 添加调试信息
#         for i, (worker, queue) in enumerate(zip(workers, queues)):
#             load = self._calculate_worker_load(worker, queue)
#             print(f"Worker {worker.wid}: load={load}, queue_len={len(queue)}, memory_usage={worker.gpu_memory_usage():.4f}")
#         worker_queue_pairs = zip(workers, queues)
#         worker, queue = min(
#             worker_queue_pairs,
#             key=lambda x: self._calculate_worker_load(x[0], x[1])
#         )
#         print(f"Selected Worker {worker.wid}")
#         return worker, queue

#     def _sched_request(self, req: 'Request', worker: 'Worker', queue: List):
#         """调度请求到指定worker的队列"""
#         # 如果是decode请求，确保在GPU中
#         if req.counter >= 0 and req.location != 'gpu':
#             req.move_to_storage('gpu', worker.wid)
            
#         queue.append(req)
#         worker.wakeup()
#         return

#     def schedule_new_req(self, req: 'Request'):
#         """调度新请求"""
#         if req.counter < 0:
#             return self.schedule_prefill(req)
#         return self.schedule_decode(req)

#     def schedule_prefill(self, req: 'Request'):
#         """调度prefill请求"""
#         assert req.counter < 0
#         worker, queue = self._find_best_worker_and_queue(
#             self._prefill_heads, 
#             queues=self._prefill_queues
#         )
#         self._sched_request(req, worker, queue)
#         return

#     def schedule_decode(self, req: 'Request'):
#         """调度decode请求
#         考虑请求的位置状态和worker的内存情况
#         """
#         print(f"\nScheduling decode request...")
#         assert req.counter >= 0
#         if req.should_finish():
#             print("Request finished, no scheduling needed")
#             req.finish_decode()
#             return

#         worker, queue = self._find_best_worker_and_queue(
#             self._decode_heads, 
#             queues=self._decode_queues
#         )
#         print(f"Scheduling request to Worker {worker.wid}, current queue length: {len(queue)}")
#         req.wait_decode(worker.wid)
#         self._sched_request(req, worker, queue)
#         return

# def put_request(env, scheduler: 'Scheduler', delays, requests):
#     """按指定延迟投放请求"""
#     for r, delay in zip(requests, delays):
#         r.init()
#         scheduler.schedule_new_req(r)
#         yield env.timeout(delay)
#     return

# def put_request_at_time(env, scheduler: 'Scheduler', time, request: 'Request'):
#     """在指定时间投放单个请求"""
#     yield env.timeout(time)
#     request.init()
#     scheduler.schedule_new_req(request)
#     return

# def put_requests_with_interarrivals(env, scheduler: 'Scheduler', inter_arrivals, requests):
#     """按照指定的时间间隔投放请求序列"""
#     assert len(inter_arrivals) == len(requests), (
#         f"Number of requests ({len(requests)}) and inter-arrivals ({len(inter_arrivals)}) "
#         f"should be the same."
#     )
#     wake_time = 0
#     for r, ts in zip(requests, inter_arrivals):
#         if r.env is None:
#             r.env = env
#         assert r.env == env
#         wake_time += ts
#         env.process(put_request_at_time(env, scheduler, wake_time, r))
#     return


from queue import Queue
from typing import List, TYPE_CHECKING, Tuple, Union

if TYPE_CHECKING:
    from simdistserve.base.request import Request
    from simdistserve.base.worker import Worker


class Scheduler:
    def __init__(self, env, prefill_heads, decode_heads):
        self.env = env
        self._prefill_heads: 'List[Worker]' = prefill_heads
        self._prefill_queues = [i.prefill_queue for i in self._prefill_heads]
        self._decode_heads: 'List[Worker]' = decode_heads
        self._decode_queues = [i.decode_queue for i in self._decode_heads]
        pass

    @staticmethod
    # def _find_best_worker_and_queue(workers, queues) -> 'Tuple[Worker, Union[Queue, List]]':
    #     # Peak the queue to find the least loaded worker.
    #     # Assume round-robin
    #     # Add the pending tasks in prefill
    #     worker, queue = min(zip(workers, queues), key=lambda x: x[0]._prefill_ips + len(x[1]))
    #     return worker, queue
    @staticmethod
    def _find_best_worker_and_queue(workers, queues):
    # 综合考虑多个因素
        def calculate_load(worker, queue):
            return (
                worker._prefill_ips * 2.0 +  # prefill负载
                worker._decode_ips * 1.5 +   # decode负载
                len(queue) * 1.0             # 队列长度
            )
    
        worker_queue_pairs = list(zip(workers, queues))
        return min(worker_queue_pairs, 
                  key=lambda x: calculate_load(x[0], x[1]))

    @staticmethod
    def _sched_request(req, worker, queue):
        queue.append(req)
        worker.wakeup()
        return

    def schedule_new_req(self, req: 'Request'):
        if req.counter < 0:
            return self.schedule_prefill(req)
        # This is for the 'decode-only' case.
        return self.schedule_decode(req)

    def schedule_prefill(self, req: 'Request'):
        assert req.counter < 0
        worker, queue = self._find_best_worker_and_queue(self._prefill_heads, queues=self._prefill_queues)
        self._sched_request(req, worker, queue)
        return
    def schedule_decode(self, req: 'Request'):
        assert req.counter >= 0
        if req.should_finish():
            print(f"Request finished, skipping scheduling")
            req.finish_decode()
            return

        worker, queue = self._find_best_worker_and_queue(
            self._decode_heads, 
            queues=self._decode_queues
        )
        print(f"Scheduling decode request to Worker {worker.wid}")
        print(f"Current worker loads:")
        for w, q in zip(self._decode_heads, self._decode_queues):
            print(f"  Worker {w.wid}: prefill_ips={w._prefill_ips}, queue_len={len(q)}")
    
        req.wait_decode(worker.wid)
        self._sched_request(req, worker, queue)

    # def schedule_decode(self, req: 'Request'):
    #     assert req.counter >= 0
    #     if req.should_finish():
    #         print(f"Request finished, skipping scheduling")
    #         # Force request to quit.
    #         req.finish_decode()
    #         return

    #     worker, queue = self._find_best_worker_and_queue(self._decode_heads, queues=self._decode_queues)
    #     req.wait_decode(worker.wid) # Artifact to prevent request having FTL != 0 when decode only.
    #     self._sched_request(req, worker, queue)
    #     return

    # pass


def put_request(env, scheduler: 'Scheduler', delays, requests):
    for r, delay in zip(requests, delays):
        r.init()
        scheduler.schedule_new_req(r)
        yield env.timeout(delay)
    return


def put_request_at_time(env, scheduler: 'Scheduler', time, request: 'Request'):
    yield env.timeout(time)
    request.init()
    scheduler.schedule_new_req(request)
    return


def put_requests_with_interarrivals(env, scheduler: 'Scheduler', inter_arrivals, requests):
    """Put requests with the inter-arrivals."""
    assert len(inter_arrivals) == len(requests), (
        f"Number of requests ({len(requests)}) and inter-arrivals ({len(inter_arrivals)}) "
        f"should be the same."
    )
    wake_time = 0
    for r, ts in zip(requests, inter_arrivals):
        if r.env is None:
            r.env = env
        assert r.env == env
        wake_time += ts
        env.process(put_request_at_time(env, scheduler, wake_time, r))
    return