def injection(self, queue, known_gaps, gap_pids, left_offset=0):
    consumed_gaps = []
    relative_consumed_gaps = []
    data_copy = queue.copy()
    for i, (payload, stride) in enumerate(data_copy):
        if len(known_gaps) > 0:
            gap = known_gaps.pop()
            if gap >= self.bitbuffer.data_size:
                exit()
            relative_consumed_gaps.append(gap)
            gap += left_offset
            consumed_gaps.append(gap)
            self.pid_list.append((gap, gap_pids[i]))
            assert stride == len(payload) / self.bitbuffer.bitsforbits * 8
            self.actual_data_hook(payload, gap, stride)
        else:
            break
    return relative_consumed_gaps, consumed_gaps, queue
