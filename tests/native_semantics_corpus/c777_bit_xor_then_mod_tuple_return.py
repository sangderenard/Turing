def fnv_step(high, low, byte):
    mixed_low = low ^ byte
    low_product = mixed_low * 435
    next_low = low_product % 4294967296
    carry = low_product // 4294967296
    next_high = (high * 435 + carry) % 4294967296
    return next_high, next_low

def root(domain, n):
    first_high, first_low = fnv_step(3, 5, domain)
    index = 0
    while index < n:
        first_high, first_low = fnv_step(first_high, first_low, index)
        index += 1
    return first_high, first_low

# INPUTS [{'domain': 1, 'n': 3}]
