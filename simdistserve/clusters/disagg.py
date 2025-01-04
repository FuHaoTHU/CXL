from functools import reduce
from itertools import chain
from typing import List, Optional

from simdistserve.base.scheduler import Scheduler
from simdistserve.base.worker import Worker, WorkerConfig
from simdistserve.utils import set_next_worker


class DisaggCluster:
    def __init__(
        self,
        env,
        N_prefill_instance: int = 1,  
        N_decode_instance: int = 1,  
        PP_prefill: int = 1,        
        PP_decode: int = 1,        
        worker_configs: 'Optional[WorkerConfig]' = None,  
        
        offload_type: str = None,     
        memory_threshold: float = 0.8,  
        gpu_memory_size: int = 32 * 1024,   
        cxl_memory_size: int = 700 * 1024, 
        local_memory_size: int = 64 * 1024, 
        cxl_load_time_per_mb: float = 0.0078125,            
        local_load_time_per_mb: float = 0.03125,           
    ):
        
        
        #######Modified#####
        self.current_TPOP = 0
        self.offload_type = offload_type
        self.memory_threshold = memory_threshold
        self.gpu_memory_size = gpu_memory_size
        self.cxl_memory_size = cxl_memory_size
        self.local_memory_size = local_memory_size


        self.prefill_instances = [] 
        self.decode_instances = []  
        worker_id = 0         
       
        worker_kwargs = dict(
            global_scheduler=None,
            offload_type=offload_type,
            memory_threshold=memory_threshold,
            cxl_load_time_per_mb=cxl_load_time_per_mb,
            local_load_time_per_mb=local_load_time_per_mb,
            gpu_memory_size=gpu_memory_size,
            cxl_memory_size=cxl_memory_size,
            local_memory_size=local_memory_size,
            **(worker_configs or {})
        )
        #####################################################
        # 1
        print("\nCreating prefill instances:")
        for inst_id in range(N_prefill_instance):
            instance = []
            
            for i in range(PP_prefill):
                worker = Worker(
                    env, 
                    worker_id, 
                    cluster=self, 
                    pipe_rank=i, 
                    **worker_kwargs
                )
                print(f"Created prefill worker {worker_id}")
                instance.append(worker)
                worker_id += 1
            
           
            reduce(set_next_worker, chain(instance, (instance[0],)))
            
            instance[-1].is_last_in_pipeline = True
            instance[-1].should_request_stay = False
            self.prefill_instances.append(instance)

        # 2
        print("\nCreating decode instances:")
        for inst_id in range(N_decode_instance):
            instance = []
            for i in range(PP_decode):
                worker = Worker(
                    env, 
                    worker_id, 
                    cluster=self, 
                    pipe_rank=i, 
                    **worker_kwargs
                )
                print(f"Created decode worker {worker_id}")
                instance.append(worker)
                worker_id += 1
            
            reduce(set_next_worker, chain(instance, (instance[0],)))
            instance[-1].is_last_in_pipeline = True
            self.decode_instances.append(instance)
        print("\nInitializing scheduler:")
        print("Prefill instances:", [[w.wid for w in inst] for inst in self.prefill_instances])
        print("Decode instances:", [[w.wid for w in inst] for inst in self.decode_instances])
        print("Prefill heads:", [inst[0].wid for inst in self.prefill_instances])
        print("Decode heads:", [inst[0].wid for inst in self.decode_instances])

        # 3
        scheduler = Scheduler(
            env,
            prefill_heads=[i[0] for i in self.prefill_instances], 
            decode_heads=[i[0] for i in self.decode_instances]
        )
        print("Created scheduler, setting up worker references...")
        
        for worker in self.get_all_workers():
            print(f"Setting global scheduler for worker {worker.wid}")
            worker.global_scheduler = scheduler

        # 4
        for last_in_prefill in (inst[-1] for inst in self.prefill_instances): 
            last_in_prefill.global_scheduler = scheduler
            print(f"Set global scheduler for last prefill worker {last_in_prefill.wid}")

        # 5
        self.env = env
        self.PP_prefill = PP_prefill
        self.PP_decode = PP_decode
        self.prefill_instances = self.prefill_instances
        self.decode_instances = self.decode_instances
        self.scheduler = scheduler

        print("Cluster initialization complete")



############################################
    # def find_optimal_threshold(self):
    #     """使用二分查找找到最优的GPU内存使用率阈值（仅CXL方案使用）"""
    #     print("\nStarting optimal threshold search")
    #     if self.offload_type != 'cxl':
    #         print("Not CXL mode, returning default threshold")
    #         return self.memory_threshold

    #     left, right = 0.3, 1.0  # 搜索范围：30%到100%
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
    #     """评估特定阈值下的性能"""
    #     # 设置新阈值
    #     print(f"\nEvaluating threshold: {threshold}")
        
        
    
    # # 更新所有worker的阈值
    #     self.memory_threshold = threshold
    #     for worker in self.get_all_workers():
    #         worker.memory_threshold = threshold
    #         self.env.process(worker.run()) ###################
    #         # worker.run()   #运行

    #     self.env.run(until=self.env.now + 1000)
            

    # # 计算并返回性能分数
    #     performance, TPOP, Total_offload_amount, Total_load_amount = self.calculate_performance()
    #     print(f"Evaluation results for threshold {threshold}:")
    #     print(f"- Throughput: {performance}")
    #     print(f"- TPOP: {TPOP}")
            
    #     return performance, TPOP   ##############  To Be Modified ##############

    
    # def calculate_performance(self):
    #     """计算当前配置下的性能分数
    # 主要关注吞吐量：生成的总token数/总时间
    #     """
    #     print(f"\nCalculating performance at simulation time: {self.env.now}")
    #     total_tokens = 0
    #     TPOP = 0
    #     count = 0
    #     total_offload_amount = 0
    #     total_load_amount = 0
    #     # 展平 decode_instances 列表
    #     decode_workers = set(worker for instance in self.decode_instances for worker in instance)
    #     print("Checking all workers status:")
    #     for worker in self.get_all_workers():
    #     # 统计所有完成的请求生成的token数、TPOP
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
    #     # 计算吞吐量：tokens/s
    #     throughput = total_tokens / self.env.now if self.env.now > 0 else 0
    #     print(f"- Throughput: {throughput}")
    #     print(f"- Average TPOP: {avg_TPOP}")
    #     return throughput, avg_TPOP, total_offload_amount, total_load_amount         #################### Modified ############################
###################################################
    def get_all_workers(self):
        
        return list(
            chain(
                chain(*self.prefill_instances),  
                chain(*self.decode_instances), 
            )
        )


# ###############################
    def run(self):
#         """修改后的运行方法"""
#         print(f"Starting cluster run with offload_type: {self.offload_type}")
#         print("Starting simulation...")
#         for instance in chain(self.prefill_instances, self.decode_instances):
#                 for worker in instance:
#                     self.env.process(worker.run())
#             # 如果是CXL方案，先找到最优阈值
#         self.env.run(until=1000)  # 先运行一段时间######
#         if self.offload_type == 'cxl':
#             # 在模拟运行过程中进行阈值搜索
#             print("\nStarting CXL threshold optimization...")
#             optimal_threshold, self.current_TPOP = self.find_optimal_threshold()
#             print(f"Optimization completed. Setting optimal threshold: {optimal_threshold}")

#             # 更新所有worker的阈值
#             self.memory_threshold = optimal_threshold
#             print("\nInitial cluster state:")
#             print(f"Number of prefill workers: {sum(len(inst) for inst in self.prefill_instances)}")
#             print(f"Number of decode workers: {sum(len(inst) for inst in self.decode_instances)}")
#             for worker in self.get_all_workers():
#                 worker.memory_threshold = optimal_threshold
#                 print(f"Updated worker {worker.wid} threshold to {optimal_threshold}")
#             self.env.process(worker.run())
#         else:
#         # 非CXL模式，直接运行到结束
#             self.env.process(worker.run())


#         return self




    
        for instance in chain(self.prefill_instances, self.decode_instances):
            for worker in instance:
                self.env.process(worker.run())
                self.env.process(worker.run_offload())
                self.env.process(worker.run_load())
        return self
###############################
