from copy import deepcopy
from collections import OrderedDict 


def hook_model(hook: dict, inter_vars: list, model, *input, **kwargs):
    # 从模型 forward 的返回值中获取中间变量, 然后返回原来的返回值
    if model is None:
        return None
    if hook is None:
        return model(*input, **kwargs)
    def get_hook_no(hook: dict, no_L: list=[]):
        for k, v in hook.items():
            if type(v) in {dict, OrderedDict}:
                get_hook_no(v, no_L)
            else:
                no_L.append(v)
        return no_L
    hook_new = {}
    # model's return: (num,Tensor,..,is tuple?,Tensor,..)
    out = model(*input, **kwargs, hook=hook_new)
    hook.update(hook_offset(hook_new, len(inter_vars)))
    inter_vars += out[1: out[0] + 1]
    # check
    no_L = sorted(get_hook_no(hook))
    assert 0 <= no_L[0] <= no_L[-1] < len(inter_vars), f'0 <= {no_L[0]} <= {no_L[-1]} < {len(inter_vars)}'
    return tuple(out[out[0] + 2:]) if out[out[0] + 1] or len(out[out[0] + 2:]) > 1 else out[out[0] + 2]


def hook_offset(hook: dict, offset: int):
    # hook 配合 inter_vars 的修改进行偏移
    for k, v in hook.items():
        if type(v) in {dict, OrderedDict}:
            hook_offset(v, offset)
        else:
            hook[k] += offset
    return hook


def hook_add(hook: dict, inter_vars: list, name, tensor):
    # hook 加入一个 tensor
    if hook is None:
        return False
    else:
        hook[name] = len(inter_vars)
        inter_vars.append(tensor)
        return True


def hook_return(hook, inter_vars, output):
    # 根据 hook 构建 model 的返回结果
    if hook is None:
        return output
    if type(output) in {list, tuple}:
        return len(inter_vars), *inter_vars, True, *output
    else:
        return len(inter_vars), *inter_vars, False, output


def hook_reduce(hook: dict, inter_vars: list, offset=0, filter=lambda t: t, root=True):
    # 将 hook 中模型的所有中间变量和名称对应起来
    hook_vars = deepcopy(hook) if root else hook
    if filter is None:
        filter = lambda t: (
            list(t.shape), t.type(), str(type(t.grad_fn)).split("'")[1]
        ) if hasattr(t, 'distill') and t.distill else None
    for k, v in list(hook.items()):
        if type(v) in {dict, OrderedDict}:
            ret = hook_reduce(v, inter_vars, offset, filter, False)
            if len(ret) == 0:
                del hook_vars[k]
            else:
                hook_vars[k] = ret
        else:
            ret = filter(inter_vars[offset + v])
            if ret is None:
                del hook_vars[k]
            else:
                hook_vars[k] = ret
    return hook_vars
