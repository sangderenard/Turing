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
        elif method_index == 4:
            inventory.call(4, memory, count)
        elif method_index == 5:
            inventory.call(5, memory, count)
        elif method_index == 6:
            inventory.call(6, memory, count)
        elif method_index == 7:
            inventory.call(7, memory, count)
        elif method_index == 8:
            inventory.call(8, memory, count)
        elif method_index == 9:
            inventory.call(9, memory, count)
        elif method_index == 10:
            inventory.call(10, memory, count)
        elif method_index == 11:
            inventory.call(11, memory, count)
        elif method_index == 12:
            inventory.call(12, memory, count)
        elif method_index == 13:
            inventory.call(13, memory, count)
