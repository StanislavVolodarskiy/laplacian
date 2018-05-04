
def pairs(seq):
    it = iter(seq)
    try:
        first = it.next()
    except StopIteration:
        return
    prev = first
    for item in it:
        yield prev, item
        prev = item
    yield prev, first
    

def triples(seq):
    it = iter(seq)
    try:
        first = it.next()
    except StopIteration:
        return
    try:
        second = it.next()
    except StopIteration:
        yield first, first, first
        return
    prev_prev = first
    prev = second
    for item in it:
        yield prev_prev, prev, item
        prev_prev, prev = prev, item
    yield prev_prev, prev, first
    yield prev, first, second
