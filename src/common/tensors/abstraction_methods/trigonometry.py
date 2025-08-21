from __future__ import annotations


def sin(self) -> "AbstractTensor":
    return self._apply_operator("sin", self, None)


def cos(self) -> "AbstractTensor":
    return self._apply_operator("cos", self, None)


def tan(self) -> "AbstractTensor":
    return self._apply_operator("tan", self, None)


def asin(self) -> "AbstractTensor":
    return self._apply_operator("asin", self, None)


def acos(self) -> "AbstractTensor":
    return self._apply_operator("acos", self, None)


def atan(self) -> "AbstractTensor":
    return self._apply_operator("atan", self, None)


def sinh(self) -> "AbstractTensor":
    return self._apply_operator("sinh", self, None)


def cosh(self) -> "AbstractTensor":
    return self._apply_operator("cosh", self, None)


def tanh(self) -> "AbstractTensor":
    return self._apply_operator("tanh", self, None)


def asinh(self) -> "AbstractTensor":
    return self._apply_operator("asinh", self, None)


def acosh(self) -> "AbstractTensor":
    return self._apply_operator("acosh", self, None)


def atanh(self) -> "AbstractTensor":
    return self._apply_operator("atanh", self, None)


# Derived via identities

def sec(self) -> "AbstractTensor":
    return self.cos() ** -1


def csc(self) -> "AbstractTensor":
    return self.sin() ** -1


def cot(self) -> "AbstractTensor":
    return self.cos() / self.sin()


def sech(self) -> "AbstractTensor":
    return self.cosh() ** -1


def csch(self) -> "AbstractTensor":
    return self.sinh() ** -1


def coth(self) -> "AbstractTensor":
    return self.cosh() / self.sinh()


def sinc(self) -> "AbstractTensor":
    return self.sin() / self

