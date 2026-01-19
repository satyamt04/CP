from __future__ import annotations
import sys
from collections import deque, defaultdict, Counter
from itertools import accumulate
from math import gcd, sqrt, ceil, floor
import math

# If recursion is used (DFS on large trees), bump the limit:
sys.setrecursionlimit(1 << 25)

data = sys.stdin.buffer.read().split()
# with open("input.txt", "rb") as f:
#     data = f.read().split()

it = iter(data)

def nstr() -> str:
    return next(it).decode()

def nint() -> int:
    return int(next(it))

def nints(k: int) -> list[int]:
    return [int(next(it)) for _ in range(k)]

def nlst(n: int, cast=int) -> list:
    return [cast(next(it)) for _ in range(n)]

def write(s: str) -> None:
    sys.stdout.write(s)

# -------------------- Debug (optional) --------------------
# Set DEBUG True for local prints; disabled on CF to avoid TLE/WA noise.
DEBUG = False
def dbg(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs, file=sys.stderr)

# -------------------- Math / Mod Utilities --------------------
MOD = 10**9 + 7 # change per problem if needed
MOD2 = 998244353

def addmod(a, b, mod=MOD): return (a + b) % mod
def submod(a, b, mod=MOD): return (a - b) % mod
def mulmod(a, b, mod=MOD): return (a * b) % mod
def powmod(a, e, mod=MOD): return pow(a, e, mod)
def invmod(a, mod=MOD): return pow(a, mod - 2, mod) # mod prime

def ceil_div(a: int, b: int) -> int:
    # Ceil division for ints with correct negatives handling
    return -(-a // b)

def lcm(a: int, b: int) -> int:
    return a // gcd(a, b) * b

# Precompute factorials (enable if needed)
def precompute_fact(n: int, mod=MOD):
    fact = [1] * (n + 1)
    invf = [1] * (n + 1)
    for i in range(1, n + 1):
        fact[i] = fact[i - 1] * i % mod
    invf[n] = pow(fact[n], mod - 2, mod)
    for i in range(n, 0, -1):
        invf[i - 1] = invf[i] * i % mod
    return fact, invf

def nCr_mod(n: int, r: int, fact, invf, mod=MOD) -> int:
    if r < 0 or r > n: return 0
    return fact[n] * invf[r] % mod * invf[n - r] % mod

# -------------------- Common Patterns --------------------
def prefix_sums(a: list[int]) -> list[int]:
    # ps[i] = sum of a[:i], ps length = len(a)+1
    return [0] + list(accumulate(a))

def bin_search_first_true(lo: int, hi: int, pred) -> int:
    """Find smallest x in [lo, hi] such that pred(x) is True. If none, returns hi+1."""
    ans = hi + 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if pred(mid):
            ans = mid
            hi = mid - 1
        else:
            lo = mid + 1
    return ans

def compress_coords(arr: list[int]):
    """Return mapping value->index (0-based), and the sorted unique list."""
    uniq = sorted(set(arr))
    rank = {v: i for i, v in enumerate(uniq)}
    return rank, uniq

# -------------------- Graph Helpers --------------------
DIR4 = [(1,0), (-1,0), (0,1), (0,-1)]
DIR8 = [(1,0), (-1,0), (0,1), (0,-1), (1,1), (1,-1), (-1,1), (-1,-1)]

def bfs_grid(starts: list[tuple[int,int]], passable) -> dict[tuple[int,int], int]:
    """Generic BFS on grid. `passable(r, c) -> bool` controls walls."""
    dq = deque()
    dist = {}
    for s in starts:
        dq.append(s)
        dist[s] = 0
    while dq:
        r, c = dq.popleft()
        for dr, dc in DIR4:
            nr, nc = r + dr, c + dc
            if (nr, nc) not in dist and passable(nr, nc):
                dist[(nr, nc)] = dist[(r, c)] + 1
                dq.append((nr, nc))
    return dist

# -------------------- DSU / Union-Find --------------------
class DSU:
    __slots__ = ("p", "r", "comp")
    def __init__(self, n: int):
        self.p = list(range(n))
        self.r = [0] * n
        self.comp = n

    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: int, b: int) -> bool:
        a, b = self.find(a), self.find(b)
        if a == b: return False
        if self.r[a] < self.r[b]:
            a, b = b, a
        self.p[b] = a
        if self.r[a] == self.r[b]:
            self.r[a] += 1
        self.comp -= 1
        return True

# ---------------------------------------- Main Solution ----------------------------------------


def solve_one():
    """
    Write your per-test-case solution here.
    Use nint(), nstr(), nlst(n), etc. to read input quickly.
    """
    a,b,c = nlst(3)

    for i in range(c//a+1):
        if (c-a*i)%b==0:
            write("YES\n")
            return
    write("NO\n")

def main(): 
    T = 1
    for _ in range(T):
        solve_one()

if __name__ == "__main__":
    main()