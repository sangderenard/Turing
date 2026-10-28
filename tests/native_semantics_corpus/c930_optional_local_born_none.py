# An Optional local declared and born as None, rebound in one arm: the
# declaration names two classes.  The annotation reader posted the present
# arm (int) as the class of the None literal and topology reduction refused
# (bootstrap: IdentityBook._post, minted: int | None = None).
def root(n: int) -> int:
    minted: int | None = None
    if n > 2:
        minted = n * 3
    if minted is None:
        return -1
    return minted

# INPUTS [{'n': 0}, {'n': 4}]
