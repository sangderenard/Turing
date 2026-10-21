import math

def root(x: float) -> float:
    return math.sqrt(x * x + 1.0) + math.sin(x) * math.cos(x) + math.exp(-x) + math.log(x + 2.0)
# INPUTS [{'x':0.5},{'x':1.25}]
