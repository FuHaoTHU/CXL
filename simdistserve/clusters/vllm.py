from functools import reduce
from itertools import chain
from typing import Optional

from simdistserve.base.scheduler import Scheduler
from simdistserve.base.worker import Worker, WorkerConfig
from simdistserve.utils import set_next_worker


class VLLMCluster:
    def __init__(
        self,
        env,
        N_instance: int = 1,
        PP: int = 1,
        model_size: str = "13b",  # 添加模型大小参数
        engine_type: str = "distserve",  # 添加引擎类型
        offload_type: Optional[str] = None,  # 添加卸载类型
        worker_configs: 'Optional[WorkerConfig]' = None,
    ):
        # 基础配置
        base_config = {
            "model_type": f"opt_{model_size}",
            "engine_type": engine_type,
            "offload_type": offload_type,
        }
        
        # 合并用户配置
        worker_kwargs = {
            **base_config,
            **(worker_configs or {}),
            "global_scheduler": None,
        }

        instances = []
        worker_id = 0
        
        # 创建实例和worker
        for inst_id in range(N_instance):
            instance = []
            for i, p in enumerate(range(PP)):
                worker = Worker(
                    env, 
                    worker_id,
                    cluster=self,
                    pipe_rank=i,
                    **worker_kwargs
                )
                instance.append(worker)
                worker_id += 1

            # 保持原有的worker链接逻辑
            reduce(
                set_next_worker,
                chain(instance, (instance[0],))
            )
            instance[-1].is_last_in_pipeline = True
            instances.append(instance)

        self.env = env
        self.PP_prefill = PP
        self.PP_decode = PP
        self.instances = instances
        self.model_size = model_size
        self.engine_type = engine_type
        self.offload_type = offload_type
        
        # 创建调度器
        self.scheduler = Scheduler(
            env, 
            prefill_heads=[i[0] for i in instances],
            decode_heads=[i[0] for i in instances]
        )

    def get_all_workers(self):
        """返回所有worker节点列表"""
        return list(chain.from_iterable(self.instances))

    def get_cluster_stats(self):
        """获取集群统计信息"""
        stats = {
            "model_size": self.model_size,
            "engine_type": self.engine_type,
            "offload_type": self.offload_type,
            "total_workers": len(self.get_all_workers()),
            "memory_usage": {},
            "throughput": {},
            "offload_amounts": {},
            "load_amounts": {},
            "total_tokens": {}
        }
        
        # 收集所有worker的统计信息
        for worker in self.get_all_workers():
            wid = worker.wid
            stats["memory_usage"][wid] = worker.gpu_memory_usage()
            stats["throughput"][wid] = worker.TPOP
            stats["offload_amounts"][wid] = worker.stats['offload_amounts']
            stats["load_amounts"][wid] = worker.stats['load_amounts']
            stats["total_tokens"][wid] = worker.stats['total_tokens']
            
        return stats

    def run(self):
        """启动集群运行"""
        for instance in self.instances:
            for worker in instance:
                self.env.process(worker.run())
        return self