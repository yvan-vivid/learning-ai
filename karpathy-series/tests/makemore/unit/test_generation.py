from typing import Tuple

from karpathy_series.makemore.models.generation import bi_gram_generate, n_gram_generate, tri_gram_generate


def forward_bigram(iv: str) -> str:
    return ({".": "h", "h": "e", "e": "l", "l": "p", "p": "."}).get(iv, "")


def forward_trigram(iv: Tuple[str, str]) -> str:
    return (
        {
            (".", "."): "h",
            (".", "h"): "e",
            ("h", "e"): "l",
            ("e", "l"): "l",
            ("l", "l"): "o",
            ("l", "o"): ".",
        }
    ).get(iv, "")


def forward_ngram(iv: str) -> str:
    return ({"...": "h", "..h": "e", ".he": "l", "hel": "l", "ell": "o", "llo": "."}).get(iv, "")


def test_generation() -> None:
    assert bi_gram_generate(forward_bigram, ".", ".") == "help"
    assert tri_gram_generate(forward_trigram, (".", "."), ".") == "hello"
    assert n_gram_generate(forward_ngram, "...", ".") == "hello"
