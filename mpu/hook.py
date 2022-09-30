# forward hook

from copy import deepcopy
from collections import OrderedDict 


def hook_model(hook: dict, inter_vars: list, model, *inputs, **kwargs):
    # 从模型 forward 的返回值中获取中间变量, 然后返回原来的返回值
    if model is None:
        return None
    if hook is None:
        return model(*inputs, **kwargs)
    def get_hook_no(hook: dict, no_L: list=[]):
        for k, v in hook.items():
            if type(v) in {dict, OrderedDict}:
                get_hook_no(v, no_L)
            elif v is not None:
                no_L.append(v)
        return no_L
    hook_new = hook_copy(hook)
    # model's return: (num,Tensor,..,is tuple?,Tensor,..)
    out = model(*inputs, **kwargs, hook=hook_new)
    hook_offset(hook_new, len(inter_vars), hook)
    inter_vars += out[1: out[0] + 1]
    # check
    no_L = sorted(get_hook_no(hook))
    if no_L:
        assert 0 <= no_L[0] <= no_L[-1] < len(inter_vars), f'0 <= {no_L[0]} <= {no_L[-1]} < {len(inter_vars)}'
    else:
        assert len(inter_vars) == 0, f'{len(inter_vars)} == 0'
    return tuple(out[out[0] + 2:]) if out[out[0] + 1] or len(out[out[0] + 2:]) > 1 else out[out[0] + 2]


def hook_offset(hook: dict, offset: int, hook_orgin=None):
    # hook 配合 inter_vars 的修改进行偏移
    for k, v in hook.items():
        if type(v) in {dict, OrderedDict}:
            hook_offset(v, offset, hook_orgin[k] if hook_orgin else None)
        elif v is not None:
            hook[k] += offset
            if hook_orgin:
                hook_orgin[k] = hook[k]
    return hook_orgin if hook_orgin else hook


def hook_copy(hook: dict, reset_no=True, root=True):
    # 深度拷贝一份 hook, v值可以置 None 便于 hook_offset 合并改变值
    if root:
        hook = deepcopy(hook)
    if not reset_no:
        return hook
    for k, v in hook.items():
        if type(v) in {dict, OrderedDict}:
            hook_copy(v, reset_no, False)
        else:
            hook[k] = None
    return hook


def hook_child(hook, name):
    # 获取子 hook 用于搜集模型的内部模型产生的 hook
    if hook is None or name not in hook:
        return None
    return hook[name]
    

def hook_add(hook: dict, inter_vars: list, name, tensor):
    # hook 加入一个 tensor
    if hook is None:
        return False
    elif name in hook:
        hook[name] = len(inter_vars)
        inter_vars.append(tensor)
        return True
    return False


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
        def filter_f(t):
            if hasattr(t, 'distill') and t.distill:
                ret = [list(t.shape), t.type(), str(type(t.grad_fn)).split("'")[1]]
                try:
                    ret.append('%e'%t.mean().item())
                except:
                    ret.append(None)
                return tuple(ret)
            else:
                return None
        filter = filter_f
    for k, v in list(hook_vars.items()):
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
