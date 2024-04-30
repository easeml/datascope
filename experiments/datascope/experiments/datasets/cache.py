import functools
import inspect
import os

from joblib import Memory
from typing import Optional, Callable, List

DEFAULT_MEMORY_CACHE_DIR = os.path.join("var", "data", "applycache")


MEMORY = Memory(DEFAULT_MEMORY_CACHE_DIR, verbose=0)


def cache(
    func: Optional[Callable] = None,
    memory: Optional[Memory] = None,
    prehash: Optional[List[str]] = None,
    ignore: Optional[List[str]] = None,
) -> Callable:
    # If the function was called as a decorator, we return a partial decorating function.
    if func is None:
        return functools.partial(cache, memory=memory, prehash=prehash, ignore=ignore)

    # By default we instantiate a default memory object.
    if memory is None:
        memory = Memory()

    # We combine the ignore list with the list of arguments to prehash.
    prehash = [] if prehash is None else prehash
    ignore = ([] if ignore is None else ignore) + prehash
    cached_func = memory.cache(func=func, ignore=ignore)

    # Extract arguments and default values.
    arg_sig = inspect.signature(func)
    arg_names = []
    arg_defaults = []
    for param in arg_sig.parameters.values():
        if param.kind is param.POSITIONAL_OR_KEYWORD or param.kind is param.KEYWORD_ONLY:
            arg_names.append(param.name)
            default = param.default if param.default is not param.empty else None
            arg_defaults.append(default)

    def wrapper(*args, **kwargs):
        # Handle arguments that we need to prehash.
        args = list(args)
        targets = {}
        for arg_position, arg_name in enumerate(arg_names):
            if arg_name in prehash:
                if arg_position < len(args):
                    targets[arg_name] = args[arg_position]
                elif arg_name in kwargs:
                    targets[arg_name] = kwargs[arg_name]
                elif arg_defaults[arg_position] is not None:
                    targets[arg_name] = arg_defaults[arg_position]
        for name, value in targets.items():
            prehash_op = getattr(value, "__hash_string__", None)
            if callable(prehash_op):
                key = "_%s_hash" % name
                kwargs[key] = prehash_op()

        return cached_func(*args, **kwargs)

    return wrapper
