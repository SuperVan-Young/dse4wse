from typing import Tuple, List, Dict
from functools import reduce
from .tensor_info import onnx_dtype_2_storage_size
from .logger import logger

class TrainingConfig():

    def __init__(self,
                 optimizer: str = "Adam",
                 precision: int = 10,  # BF16 
                 mix_precision: bool = True,  # use FP32 for op_state
                 activation_checkpoint: Dict[str, bool] = {},  # 
                 ) -> None:
        self.optimizer = optimizer
        self.precision = precision
        self.mix_precision = mix_precision
        self.activation_checkpoint = activation_checkpoint
        #FIXME:  adjust precision when building the graph

    def get_precision_size(self) -> int:
        return onnx_dtype_2_storage_size(self.precision)

    def get_dynamic_optimizer_state_size(self) -> int:
        """Updating one weight on SRAM requires this amount of temporary buffer.
        """
        logger.warning(f"{__name__}: Deprecated in the future")
        update_precision = 4 if self.mix_precision else onnx_dtype_2_storage_size(self.precision)
        if self.optimizer == "Adam":
            return update_precision
        else:
            raise NotImplementedError

    def get_static_optimizer_state_size(self) -> int:
        """These optimizer state need to be stored on DRAM throughout training one weight,
        and loaded to SRAM when necessary.
        """
        logger.warning(f"{__name__}: Deprecated in the future")
        update_precision = 4 if self.mix_precision else onnx_dtype_2_storage_size(self.precision)
        if self.optimizer == "Adam":
            return update_precision * 2  #  momentum + variance
        else:
            raise NotImplementedError
        
    def need_rematerialization(self, tensor_name: str) -> bool:
        return self.activation_checkpoint.get(tensor_name, True)
    
    def get_optimizer_state_size(self) -> int:
        update_precision = 4 if self.mix_precision else onnx_dtype_2_storage_size(self.precision)
        if self.optimizer == "Adam":
            return update_precision * 3  #  momentum + variance + grad
        else:
            raise NotImplementedError
        
    def get_weight_update_compute_amount(self) -> int:
        if self.optimizer == 'Adam':
            m_t = 5
            v_t = 6
            w = 4 + 4 # sqrt is approximately 4~13 MAC
            return m_t + v_t + w
        else:
            raise NotADirectoryError