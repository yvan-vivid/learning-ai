from functools import partial
from typing import Callable, Tuple, TypeVar

V = TypeVar("V")


def generate_string(
    feedback: Callable[[V], Callable[[str], V]],
    forward: Callable[[V], str],
    initial: V,
    final: str,
    max_length: int = 100,
) -> str:
    fb_state = feedback(in_v := initial)
    out = ""
    for _k in range(max_length):
        if (out_v := forward(in_v)) == final:
            break
        out += out_v
        in_v = fb_state(out_v)
    return out


def feedbacker(update: Callable[[V, str], V], initial: V) -> Callable[[str], V]:
    state = initial

    def _inner(c: str) -> V:
        nonlocal state
        state = update(state, c)
        return state

    return _inner


def _n_gram_update(state: str, c: str) -> str:
    return state[1:] + c


def _tri_gram_update(state: Tuple[str, str], c: str) -> Tuple[str, str]:
    return state[1], c


n_gram_generate = partial(generate_string, partial(feedbacker, _n_gram_update))
bi_gram_generate = partial(generate_string, partial(feedbacker, (lambda _state, c: c)))
tri_gram_generate = partial(generate_string, partial(feedbacker, _tri_gram_update))
