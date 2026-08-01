def coordinate_class_range(memory, inventory, count, start, end):
    for method_index in range(start, end, 1):
        if method_index == 0:
            inventory.call(0, memory, count)
        elif method_index == 1:
            inventory.call(1, memory, count)
        elif method_index == 2:
            inventory.call(2, memory, count)
        elif method_index == 3:
            inventory.call(3, memory, count)
