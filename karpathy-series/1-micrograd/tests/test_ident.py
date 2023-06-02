from micrograd.ident import IdentManager


def test_generate_incrementing():
    im = IdentManager()
    assert im.use() == 0
    assert im.use() == 1
    assert im.use() == 2


def test_labeling():
    im = IdentManager()
    l0, l1, l2, l3 = im.use(), im.use(), im.use(), im.use()
    im.set_label(l1, "l1")
    im.set_label(l2, "l2")

    assert im.label(l1) == "l1"
    assert im.label(l2) == "l2"
    assert im.label(l0) is None
    assert im.label(l3) is None

    assert im.labels() == frozenset({"l1", "l2"})
