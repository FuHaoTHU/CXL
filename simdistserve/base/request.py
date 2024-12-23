"""
Request类 - 用于模拟系统中的请求对象
"""
# 定义请求的各种事件类型
E_INIT = "init"                    # 初始化事件
E_WAIT_PREFILL = "wait_prefill"    # 等待预填充事件
E_DO_PREFILL = "do_prefill"        # 执行预填充事件
E_WAIT_DECODE = "wait_decode"      # 等待解码事件
E_DO_DECODE = "do_decode"          # 执行解码事件
E_FINISH_PREFILL = "finish_prefill"  # 完成预填充事件
E_FINISH_DECODE = "finish_decode"    # 完成解码事件
E_EXIT_SYSTEM = "exit_system"        # 退出系统事件

#####################################################
# 新增KVCache管理相关事件
E_START_OFFLOAD = "start_offload"    # 开始卸载到CXL
E_FINISH_OFFLOAD = "finish_offload"  # 完成卸载
E_START_LOAD = "start_load"          # 开始从CXL加载
E_FINISH_LOAD = "finish_load"        # 完成加载
E_START_OFFLOAD_LOCAL = "start_offload_local"
E_FINISH_OFFLOAD_LOCAL = "finish_offload_local"
E_START_LOAD_LOCAL = "start_load_local"
E_FINISH_LOAD_LOCAL = "finish_load_local"
#####################################################

import numpy as np
import simpy

class Request:
    def __str__(self):
        """请求对象的字符串表示"""
        return (
            f'Request('
            f'id={self.req_id},'
            f',prefill={self.prefill_lens}'
            f',output={self.output_lens}'
            f')'
        )

    __repr__ = __str__



    def __init__(
        self,
        env: 'simpy.Environment' = None,  # 模拟环境
        req_id: int = None,               # 请求ID
        req_init_counter=-1,              # 初始计数器
        prefill_length: int = 512,        # 预填充长度
        output_lens: int = 128000000,           # 输出长度################################设为非定值
        schedule_wait: int = 0,            # 调度等待时间
        offload_type: str = None
    ):
        """
        初始化请求对象
        
        counter说明:
        - counter < 0: 预填充步骤。通常使用-1
        - counter >= 0: 解码步骤。最大值是output_lens - 1
        """
        assert req_id is not None, f'Request ID is not set.'
        self.env = env
        self.req_id = req_id
        self.counter = req_init_counter      # 请求状态计数器
        self.log = []                        # 事件日志列表
        self._terminated = False             # 请求是否终止
        self.prefill_lens = prefill_length   # 预填充总长度
        self.output_lens = output_lens       # 需要生成的输出长度
        self.schedule_wait = schedule_wait    # 调度等待时间
        # 新增的属性#################################################    
        self.offload_type = offload_type  # 'cxl', 'local', None
        self.output_lens = output_lens
        self.location = 'gpu'  # 当前位置：'gpu' 或 'cxl' 'local'
                # 优先级计算只在CXL方案中使用
        self.priority = self._calculate_priority() if offload_type == 'cxl' else None
        ############################################################
        # 未调度的预填充长度
        self.remain_prefill_lens = prefill_length
        # 当前活跃的预填充长度
        self.current_prefill_lens = 0
        # 分块ID（用于分块预填充）
        self.chunk_id = None
        
    ###################################################
    def _generate_output_length(self) -> int:
        """生成长尾分布的输出长度"""
        # 使用对数正态分布生成长尾分布的token长度
        # mean=4 使得大多数请求长度在50-100之间
        # sigma=1 提供足够的变异性
        return int(np.random.lognormal(mean=4, sigma=1))

    def _calculate_priority(self) -> float:
        """计算请求优先级
        优先级与预期token数成反比
        添加小的epsilon避免除零
        """
        epsilon = 1e-6
        return 1.0 / (self.output_lens + epsilon)
    ##################################################




    @property
    def current_context_len(self):
        """
        计算当前上下文长度(也就时KVcache的大小)
        = 预填充长度 + 已生成的token数量
        
        解释：
        - prefill_lens: 输入序列的长度
        - counter: 如果为负数表示在预填充阶段，为正数表示已生成的token数
        - max(0, self.counter): 确保只在解码阶段增加长度
        """
        return self.prefill_lens + max(0, self.counter)
    @property
    def current_kvcache_size(self):
        return 2 * 32 * 32 * 128 * 2 * self.current_context_len


    ###################################
    @property
    def is_in_gpu(self) -> bool:
        """检查请求是否在GPU中"""
        return self.location == 'gpu'
    def move_to_storage(self, storage_type: str, wid=None):
        """统一的存储位置转移方法"""
        old_location = self.location
        self.location = storage_type
        
        # 根据不同的存储类型记录相应事件
        if storage_type != 'gpu':
            # 卸载事件
            event_prefix = 'E_START_OFFLOAD_'
            if storage_type == 'cxl':
                self._log_event(E_START_OFFLOAD, wid=wid)
            else:  # local
                self._log_event(E_START_OFFLOAD_LOCAL, wid=wid)
        else:
            # 加载事件
            if old_location == 'cxl':
                self._log_event(E_START_LOAD, wid=wid)
            else:  # local
                self._log_event(E_START_LOAD_LOCAL, wid=wid)

    ##################################





#####################################################
    def offload_to_cxl(self, wid=None):
        """记录卸载开始事件"""
        self._log_event(E_START_OFFLOAD, wid=wid)
        self.move_to_cxl()
        self._log_event(E_FINISH_OFFLOAD, wid=wid)

    def load_to_gpu(self, wid=None):
        """记录加载事件"""
        self._log_event(E_START_LOAD, wid=wid)
        self.move_to_gpu()
        self._log_event(E_FINISH_LOAD, wid=wid)
#####################################################
    def _log_event(self, event, wid=-1):
        """记录事件到日志"""
        if not self.env:
            raise ValueError("Request.env is not set.")
        if self._terminated:
            return
        self.log.append((self.env.now, event, wid))

    # 各种事件处理方法
    def init(self):
        """初始化请求"""
        self._log_event(E_INIT)

    def wait_prefill(self, wid=None):
        """等待预填充"""
        self._log_event(E_WAIT_PREFILL, wid=wid)

    def do_prefill(self, wid=None):
        """执行预填充"""
        self._log_event(E_DO_PREFILL, wid=wid)

    def wait_decode(self, wid=None):
        """等待解码"""
        self._log_event(E_WAIT_DECODE, wid=wid)

    def do_decode(self, wid=None):
        """执行解码"""
        self._log_event(E_DO_DECODE, wid=wid)

    def _reset_chunked_prefill_metadata(self):
        """重置分块预填充的元数据"""
        self.chunk_id = None
        self.current_prefill_lens = 0

    def finish_prefill(self, is_finished_one_round=False, wid=None, next_wid=None):
        """
        完成预填充处理
        
        参数:
        is_finished_one_round: 是否完成一轮预填充
        wid: 当前worker ID
        next_wid: 下一个worker ID
        """
        if not is_finished_one_round:
            self.wait_prefill(wid=next_wid)
            return

        self._reset_chunked_prefill_metadata()
        if self.remain_prefill_lens > 0:
            self.wait_prefill(wid=next_wid)
            return

        # 所有预填充完成，重置计数器为0（开始解码阶段）
        self.counter = 0
        self.wait_decode(wid=next_wid)
        if not self.should_finish():
            return
        self._log_event(E_EXIT_SYSTEM)
        self._terminated = True

    def finish_decode(self, is_finished_one_round=False, next_wid=None):
        """
        完成解码处理
        
        参数:
        is_finished_one_round: 是否完成一轮解码
        next_wid: 下一个worker ID
        """
        if is_finished_one_round:
            self.counter += 1  # 增加已生成的token计数
        self.wait_decode(wid=next_wid)
        if self.should_finish():
            self._log_event(E_EXIT_SYSTEM)
            self._terminated = True

    def should_finish(self):
        """判断请求是否应该结束（已生成足够的token）"""
        return self.counter >= self.output_lens