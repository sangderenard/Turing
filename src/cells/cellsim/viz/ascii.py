def bar_from_fractions(fracs, width, chars=None):
    chars = chars or [chr(97+i) for i in range(len(fracs))]
    segs = [int(f*width) for f in fracs]
    segs[-1] += width - sum(segs)
    return '|' + ''.join(chars[i]*segs[i] for i in range(len(fracs))) + '|'
