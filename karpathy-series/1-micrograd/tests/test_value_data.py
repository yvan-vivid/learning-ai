from micrograd.value_data import ValueData


def test_const_value_data():
    assert ValueData.const(5) == ValueData("const", 5)


def test_add_value_data():
    assert ValueData.const(5) + ValueData.const(6) == ValueData("+", 11)


def test_mult_value_data():
    assert ValueData.const(5) * ValueData.const(6) == ValueData("*", 30)


def test_neg_value_data():
    assert -ValueData.const(5) == ValueData("-", -5)
