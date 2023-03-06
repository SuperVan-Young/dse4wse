
from base import Operator
from typing import Union

def build_operator(operator_conf) -> Operator:
    """Build operator from oneflow proto.
    """
    op_type = operator_conf.WhichOneof('op_type')

    if op_type == "user_conf":
        operator = build_user_conf_operator(operator_conf)
    else:
        raise NotImplementedError

    return operator


def build_user_conf_operator(operator_conf):
    user_conf = operator_conf.user_conf
    op_type_name = user_conf.op_type_name
    