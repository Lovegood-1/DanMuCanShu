"""
探索生成器的性质
"""
from certifi import where


def gen():
    init = 0
    while init < 5:
        init += 1
        print("before yield",init)
        yield init
        print("after yield ", init)
    raise StopIteration
g = gen()
for i in range(7):
    g1 = next(g)
    print(i,g1)
a = 1