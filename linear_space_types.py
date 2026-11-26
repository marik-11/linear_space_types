import argparse
import json
import os
import random
import sys
import time
from itertools import combinations
from multiprocessing import Pool


def multisets(n):
    total = n * (n - 1) // 2
    out = []
    cur = []

    def gen(maxpart, s):
        out.append(tuple(cur))
        for k in range(min(maxpart, n), 2, -1):
            c = k * (k - 1) // 2
            if s + c > total:
                continue
            if cur and cur[0] + k > n + 1:
                continue
            cur.append(k)
            gen(k, s + c)
            cur.pop()

    gen(n, 0)
    out.sort(key=lambda m: (sum(k * (k - 1) // 2 for k in m), m))
    return out


def lower_neighbors(m):
    seen = set()
    for i in range(len(m)):
        if i and m[i] == m[i - 1]:
            continue
        d = list(m)
        if m[i] == 3:
            del d[i]
        else:
            d[i] -= 1
        d.sort(reverse=True)
        t = tuple(d)
        if t not in seen:
            seen.add(t)
            yield t


def packbound(n):
    d = n * ((n - 1) // 2) // 3
    if n % 6 == 5:
        d -= 1
    return d


def outside_ok(m, n):
    if not m:
        return True
    k = m[0]
    out = n - k
    need = 0
    weight = 0
    for j in m[1:]:
        if j - 1 > out:
            return False
        need += j * (j - 1) // 2
        weight += j - 1
    saved = min(weight, k * out)
    return need - saved <= out * (out - 1) // 2


def graphical(d):
    d = sorted(d, reverse=True)
    if sum(d) & 1:
        return False
    k = len(d)
    left = 0
    for r in range(1, k + 1):
        left += d[r - 1]
        right = r * (r - 1) + sum(min(x, r) for x in d[r:])
        if left > right:
            return False
    return True


def assignable(m, n):
    sizes = []
    cnt = []
    for k in m:
        if not sizes or sizes[-1] != k:
            sizes.append(k)
            cnt.append(0)
        cnt[-1] += 1
    types = []

    def gen(i, cur, load):
        if i == len(sizes):
            types.append(tuple(cur))
            return
        x = 0
        while x <= cnt[i] and load + x * (sizes[i] - 1) <= n - 1:
            cur.append(x)
            gen(i + 1, cur, load + x * (sizes[i] - 1))
            cur.pop()
            x += 1

    gen(0, [], 0)
    types.sort(key=lambda t: -sum(x * (s - 1) for x, s in zip(t, sizes)))
    rem = [s * c for s, c in zip(sizes, cnt)]
    degs = []
    steps = [0]

    def dfs(v, mint):
        steps[0] += 1
        if steps[0] > 2000000:
            return True
        if v == n:
            return all(r == 0 for r in rem) and graphical(degs)
        left = n - v
        for i in range(len(sizes)):
            if rem[i] > left * cnt[i]:
                return False
        for t in range(mint, len(types)):
            ty = types[t]
            if any(ty[i] > rem[i] for i in range(len(sizes))):
                continue
            load = sum(ty[i] * (sizes[i] - 1) for i in range(len(sizes)))
            for i in range(len(sizes)):
                rem[i] -= ty[i]
            degs.append(n - 1 - load)
            r = dfs(v + 1, t)
            degs.pop()
            for i in range(len(sizes)):
                rem[i] += ty[i]
            if r:
                return True
        return False

    return dfs(0, 0)


def random_phase(n, runs, seed):
    rng = random.Random(seed)
    found = set()
    full = (1 << n) - 1
    for _ in range(runs):
        un = [full & ~(1 << v) for v in range(n)]
        sizes = []
        p3 = 0.35 + 0.6 * rng.random()
        while True:
            vs = [v for v in range(n) if un[v]]
            if not vs:
                break
            u = rng.choice(vs)
            nb = [x for x in range(n) if (un[u] >> x) & 1]
            w = rng.choice(nb)
            S = [u, w]
            cand = un[u] & un[w]
            while cand:
                cs = [x for x in range(n) if (cand >> x) & 1]
                x = rng.choice(cs)
                S.append(x)
                cand &= un[x]
                if len(S) >= 3 and rng.random() < p3:
                    break
            for a in S:
                for b in S:
                    if a != b:
                        un[a] &= ~(1 << b)
            if len(S) >= 3:
                sizes.append(len(S))
        sizes.sort(reverse=True)
        found.add(tuple(sizes))
    return sorted(found, key=len, reverse=True)


def dominated(m, feas):
    lm = len(m)
    for f in feas:
        if len(f) < lm:
            break
        if all(m[i] <= f[i] for i in range(lm)):
            return True
    return False


def popcount(x):
    return bin(x).count("1")


class Search:
    def __init__(self, parts, n, cap=None):
        self.parts = parts
        self.n = n
        self.cap = cap
        self.nodes = 0
        self.aborted = False
        self.rem_inc = [0] * (len(parts) + 1)
        for i in range(len(parts) - 1, -1, -1):
            self.rem_inc[i] = self.rem_inc[i + 1] + parts[i]

    def children(self, idx, used, unc, prev_tuple):
        n = self.n
        parts = self.parts
        k = parts[idx]
        kmin = parts[-1]
        capsum = sum(popcount(u) // (kmin - 1) for u in unc)
        if capsum < self.rem_inc[idx]:
            return
        if sum(1 for u in unc if popcount(u) >= k - 1) < k:
            return
        same = idx > 0 and parts[idx] == parts[idx - 1]
        lo_fu = max(0, k - used)
        hi_fu = min(k, n - used)
        for fu in range(lo_fu, hi_fu + 1):
            need = k - fu
            start_cand = (1 << used) - 1
            stack = []

            def rec(chosen, cand, start):
                if len(chosen) == need:
                    allp = chosen + list(range(used, used + fu))
                    if same and not (allp > prev_tuple):
                        return
                    nu = list(unc)
                    for a in allp:
                        for b in allp:
                            if a != b:
                                nu[a] &= ~(1 << b)
                    yield (used + fu, nu, allp)
                    return
                for v in range(start, used):
                    if not (cand >> v) & 1:
                        continue
                    chosen.append(v)
                    yield from rec(chosen, cand & unc[v], v + 1)
                    chosen.pop()

            yield from rec([], start_cand, 0)

    def dfs(self, idx, used, unc, prev_tuple):
        self.nodes += 1
        if self.cap is not None and self.nodes > self.cap:
            self.aborted = True
            return False
        if idx == len(self.parts):
            return True
        for nused, nu, tup in self.children(idx, used, unc, prev_tuple):
            if self.dfs(idx + 1, nused, nu, tup):
                return True
            if self.aborted:
                return False
        return False


def initial_state(n):
    return 0, [((1 << n) - 1) & ~(1 << v) for v in range(n)], []


def hall_ok(vt, sizes, n):
    m = len(sizes)
    subsets = []
    for mask in range(1, 1 << m):
        subsets.append([i for i in range(m) if (mask >> i) & 1])
    for v in range(n):
        for T in subsets:
            need = sum(vt[v][i] * (sizes[i] - 1) for i in T)
            if need == 0:
                continue
            avail = 0
            for w in range(n):
                if w != v and any(vt[w][i] > 0 for i in T):
                    avail += 1
            if need > avail:
                return False
    return True


def intersection_ok(vt, sizes, cnt, n):
    m = len(sizes)
    for i in range(m):
        tot = sum(vt[v][i] * (vt[v][i] - 1) // 2 for v in range(n))
        if tot > cnt[i] * (cnt[i] - 1) // 2:
            return False
        for j in range(i + 1, m):
            cross = sum(vt[v][i] * vt[v][j] for v in range(n))
            if cross > cnt[i] * cnt[j]:
                return False
    return True


def distributions(parts, n, filters=True):
    sizes = sorted(set(parts), reverse=True)
    cnt = [parts.count(s) for s in sizes]
    m = len(sizes)
    types = []

    def gen(i, cur, load):
        if i == m:
            types.append(tuple(cur))
            return
        x = 0
        while x <= cnt[i] and load + x * (sizes[i] - 1) <= n - 1:
            cur.append(x)
            gen(i + 1, cur, load + x * (sizes[i] - 1))
            cur.pop()
            x += 1

    gen(0, [], 0)
    types.sort(key=lambda t: (-sum(x * (s - 1) for x, s in zip(t, sizes)), t))
    nt = len(types)
    tload = [sum(x * (sz - 1) for x, sz in zip(t, sizes)) for t in types]
    suffix_max = [[0] * m for _ in range(nt + 1)]
    for u in range(nt - 1, -1, -1):
        suffix_max[u] = [max(suffix_max[u + 1][i], types[u][i]) for i in range(m)]
    e_total = n * (n - 1) // 2 - sum(k * (k - 1) // 2 for k in parts)
    pair_cap = [c * (c - 1) // 2 for c in cnt]
    cross_cap = {(i, j): cnt[i] * cnt[j] for i in range(m) for j in range(i + 1, m)}
    memo = {}

    typeidx = {ty: u for u, ty in enumerate(types)}

    def reach(v_left, t, rem):
        if v_left == 0:
            return not any(rem)
        if v_left == 1:
            u = typeidx.get(rem)
            return u is not None and u >= t
        key = (v_left, t, rem)
        r = memo.get(key)
        if r is not None:
            return r
        res = False
        sm = suffix_max[t]
        ok = True
        tot = 0
        for i in range(m):
            if rem[i] > v_left * sm[i]:
                ok = False
                break
            tot += rem[i] * (sizes[i] - 1)
        if ok and tot > v_left * tload[t]:
            ok = False
        if ok:
            for u in range(t, nt):
                ty = types[u]
                good = True
                for i in range(m):
                    if ty[i] > rem[i]:
                        good = False
                        break
                if not good:
                    continue
                nrem = tuple([rem[i] - ty[i] for i in range(m)])
                if reach(v_left - 1, u, nrem):
                    res = True
                    break
        memo[key] = res
        return res

    rem0 = tuple(s * c for s, c in zip(sizes, cnt))
    chosen = []
    out = []
    raw = [0]
    pair_sum = [0] * m
    ind_sum = [0] * m
    cross_sum = {k: 0 for k in cross_cap}

    def dfs(v, mint, rem):
        if v == n:
            d = [n - 1 - sum(x * (s - 1) for x, s in zip(types[t], sizes)) for t in chosen]
            if graphical(d):
                raw[0] += 1
                vt = [types[t] for t in chosen]
                if not filters or hall_ok(vt, sizes, n):
                    out.append(list(chosen))
            return
        for t in range(mint, nt):
            ty = types[t]
            if any(ty[i] > rem[i] for i in range(m)):
                continue
            nrem = tuple(rem[i] - ty[i] for i in range(m))
            if any(nrem[i] > (n - v - 1) * suffix_max[t][i] for i in range(m)):
                continue
            if sum(nrem[i] * (sizes[i] - 1) for i in range(m)) > (n - v - 1) * tload[t]:
                continue
            if not reach(n - v - 1, t, nrem):
                continue
            if filters:
                dl_t = n - 1 - tload[t]
                bad_ind = False
                for i in range(m):
                    if ind_sum[i] + ty[i] * dl_t > cnt[i] * e_total:
                        bad_ind = True
                        break
                if bad_ind:
                    continue
            bad = False
            dp = [ty[i] * (ty[i] - 1) // 2 for i in range(m)]
            dc = {k: ty[k[0]] * ty[k[1]] for k in cross_cap}
            if filters:
                for i in range(m):
                    if pair_sum[i] + dp[i] > pair_cap[i]:
                        bad = True
                        break
                if not bad:
                    for k in cross_cap:
                        if cross_sum[k] + dc[k] > cross_cap[k]:
                            bad = True
                            break
                if bad:
                    continue
                for i in range(m):
                    pair_sum[i] += dp[i]
                    ind_sum[i] += ty[i] * (n - 1 - tload[t])
                for k in cross_cap:
                    cross_sum[k] += dc[k]
            chosen.append(t)
            dfs(v + 1, t, nrem)
            chosen.pop()
            if filters:
                for i in range(m):
                    pair_sum[i] -= dp[i]
                    ind_sum[i] -= ty[i] * (n - 1 - tload[t])
                for k in cross_cap:
                    cross_sum[k] -= dc[k]

    dfs(0, 0, rem0)
    return sizes, types, out, raw[0]



def canon_key(sizes, verts, unc, need, leftneed):
    k = len(verts)
    index = {v: i for i, v in enumerate(verts)}
    adj = [0] * k
    for i, v in enumerate(verts):
        row = unc[v]
        a = 0
        for w in verts:
            if (row >> w) & 1:
                a |= 1 << index[w]
        adj[i] = a
    attr = [(tuple(need[v]), leftneed[v]) for v in verts]
    order_attr = sorted(set(attr))
    rank = {a: i for i, a in enumerate(order_attr)}
    color = [rank[a] for a in attr]
    return _canon(k, adj, attr, color)


def _refine(k, adj, color):
    while True:
        sig = []
        for v in range(k):
            nb = sorted(color[u] for u in range(k) if (adj[v] >> u) & 1)
            sig.append((color[v], tuple(nb)))
        ranks = {sg: i for i, sg in enumerate(sorted(set(sig)))}
        new = [ranks[sg] for sg in sig]
        if len(set(new)) == len(set(color)):
            return new
        color = new


def _canon(k, adj, attr, color):
    best = [None]

    def rec(col):
        col = _refine(k, adj, col)
        classes = {}
        for v in range(k):
            classes.setdefault(col[v], []).append(v)
        target = None
        for c in sorted(classes):
            if len(classes[c]) > 1:
                target = classes[c]
                break
        if target is None:
            order = sorted(range(k), key=lambda v: col[v])
            pos = {v: i for i, v in enumerate(order)}
            cert = tuple((attr[v], tuple(sorted(pos[u] for u in range(k) if (adj[v] >> u) & 1))) for v in order)
            if best[0] is None or cert < best[0]:
                best[0] = cert
            return
        done = []
        for v in target:
            skip = False
            for u in done:
                if attr[u] == attr[v] and (adj[u] & ~(1 << v)) == (adj[v] & ~(1 << u)):
                    skip = True
                    break
            if skip:
                continue
            done.append(v)
            nc = [2 * c for c in col]
            nc[v] -= 1
            rec(nc)

    rec(color)
    return best[0]


FAILED = set()
FAILED_LIMIT = 3000000


class Typed:
    def __init__(self, sizes, vt, n, cap, look=True):
        self.n = n
        self.sizes = sizes
        self.m = len(sizes)
        self.vtypes = vt
        self.cap = cap
        self.look = look
        self.memo_min = 5
        self.nodes = 0
        self.aborted = False
        self.full = (1 << n) - 1
        self.dleft = [n - 1 - sum(x * (s - 1) for x, s in zip(t, sizes)) for t in vt]
        self.cls = []
        c = 0
        for v in range(n):
            if v and vt[v] != vt[v - 1]:
                c += 1
            self.cls.append(c)

    def local_ok(self, x, unc, have, left):
        sizes = self.sizes
        m = self.m
        need = [self.vtypes[x][i] - have[x][i] for i in range(m)]
        nl = self.dleft[x] - left[x]
        N = unc[x]
        return self.partition(x, N, unc, have, left, need, nl)

    def partition(self, x, N, unc, have, left, need, nl):
        if N == 0:
            return nl == 0 and not any(need)
        w = (N & -N).bit_length() - 1
        rest = N & ~(1 << w)
        if nl > 0 and left[w] < self.dleft[w]:
            if self.partition(x, rest, unc, have, left, need, nl - 1):
                return True
        pool = rest & unc[w]
        pc = popcount(pool)
        for i, s in enumerate(self.sizes):
            if need[i] == 0 or s - 2 > pc:
                continue
            if have[w][i] >= self.vtypes[w][i]:
                continue
            k = s - 2
            lst = [y for y in range(self.n) if (pool >> y) & 1 and have[y][i] < self.vtypes[y][i]]
            if len(lst) < k:
                continue
            for T in combinations(lst, k):
                ok = True
                for a in range(len(T)):
                    for b in range(a + 1, len(T)):
                        if not (unc[T[a]] >> T[b]) & 1:
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    continue
                newN = rest
                for y in T:
                    newN &= ~(1 << y)
                need[i] -= 1
                r = self.partition(x, newN, unc, have, left, need, nl)
                need[i] += 1
                if r:
                    return True
        return False

    def run(self):
        n = self.n
        unc = [self.full & ~(1 << v) for v in range(n)]
        have = [[0] * self.m for _ in range(n)]
        left = [0] * n
        return self.vertex(self.full, unc, have, left)

    def vertex(self, unprocessed, unc, have, left):
        self.nodes += 1
        if self.cap is not None and self.nodes > self.cap:
            self.aborted = True
            return False
        if unprocessed == 0:
            return True
        if self.look:
            mask = unprocessed
            while mask:
                x = (mask & -mask).bit_length() - 1
                mask &= mask - 1
                if not self.local_ok(x, unc, have, left):
                    return False
        memo_key = None
        if popcount(unprocessed) >= self.memo_min:
            verts = [x for x in range(self.n) if (unprocessed >> x) & 1]
            need_all = {x: [self.vtypes[x][i] - have[x][i] for i in range(self.m)] for x in verts}
            lneed = {x: self.dleft[x] - left[x] for x in verts}
            memo_key = (tuple(self.sizes), canon_key(self.sizes, verts, unc, need_all, lneed))
            if memo_key in FAILED:
                return False
        best = -1
        bestkey = None
        mask = unprocessed
        while mask:
            x = (mask & -mask).bit_length() - 1
            mask &= mask - 1
            key = (popcount(unc[x]), x)
            if bestkey is None or key < bestkey:
                bestkey = key
                best = x
        v = best
        need = [self.vtypes[v][i] - have[v][i] for i in range(self.m)]
        nl = self.dleft[v] - left[v]
        if nl < 0 or any(x < 0 for x in need):
            return False
        if popcount(unc[v]) != sum(need[i] * (self.sizes[i] - 1) for i in range(self.m)) + nl:
            return False
        rest_unprocessed = unprocessed & ~(1 << v)
        res = self.blocks(v, unc, have, left, need, nl, unc[v], unprocessed, rest_unprocessed)
        if not res and not self.aborted and memo_key is not None and len(FAILED) < FAILED_LIMIT:
            FAILED.add(memo_key)
        return res

    def blocks(self, v, unc, have, left, need, nl, A, unprocessed, rest_unprocessed):
        if A == 0:
            if nl != 0 or any(need):
                return False
            nu = list(unc)
            nu[v] = 0
            return self.vertex(rest_unprocessed, nu, have, left)
        w = (A & -A).bit_length() - 1
        rest = A & ~(1 << w)
        if nl > 0 and left[w] < self.dleft[w]:
            nu = list(unc)
            nu[v] &= ~(1 << w)
            nu[w] &= ~(1 << v)
            left[w] += 1
            ok = self.blocks(v, nu, have, left, need, nl - 1, rest, unprocessed, rest_unprocessed)
            left[w] -= 1
            if ok:
                return True
            if self.aborted:
                return False
        pool = rest & unc[w]
        pc = popcount(pool)
        for idx, s in enumerate(self.sizes):
            if need[idx] == 0 or s - 2 > pc:
                continue
            if have[w][idx] >= self.vtypes[w][idx]:
                continue
            k = s - 2
            touched = []
            groups = self.twin_classes(pool, unc, have, left, idx)
            for T in self.choose(touched, groups, k):
                members = T + [w, v]
                ok = True
                for a in range(len(members)):
                    for b in range(a + 1, len(members)):
                        if not (unc[members[a]] >> members[b]) & 1:
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    continue
                if not all(have[a][idx] < self.vtypes[a][idx] for a in members):
                    continue
                nu = list(unc)
                for a in members:
                    for b in members:
                        if a != b:
                            nu[a] &= ~(1 << b)
                for a in members:
                    have[a][idx] += 1
                nneed = list(need)
                nneed[idx] -= 1
                newA = rest
                for x in T:
                    newA &= ~(1 << x)
                ok2 = self.blocks(v, nu, have, left, nneed, nl, newA, unprocessed, rest_unprocessed)
                for a in members:
                    have[a][idx] -= 1
                if ok2:
                    return True
                if self.aborted:
                    return False
        return False

    def twin_classes(self, pool, unc, have, left, idx):
        reps = []
        classes = []
        for x in range(self.n):
            if not (pool >> x) & 1:
                continue
            placed = False
            for ci, r in enumerate(reps):
                if (self.vtypes[x] == self.vtypes[r] and have[x] == have[r] and left[x] == left[r]
                        and (unc[x] & ~(1 << r)) == (unc[r] & ~(1 << x))):
                    classes[ci].append(x)
                    placed = True
                    break
            if not placed:
                reps.append(x)
                classes.append([x])
        return {ci: lst for ci, lst in enumerate(classes)}

    def choose(self, touched, groups, k):
        group_items = list(groups.items())
        res = []

        def rec(gi, cnt_left, acc):
            if gi == len(group_items):
                if cnt_left >= 0:
                    for tc in combinations(touched, cnt_left):
                        res.append(list(tc) + acc)
                return
            cls, lst = group_items[gi]
            for j in range(0, min(cnt_left, len(lst)) + 1):
                rec(gi + 1, cnt_left - j, acc + lst[:j])

        rec(0, k, [])
        return res


def order_distributions(sizes, types, dists, n):
    def score(d):
        dl = [n - 1 - sum(x * (sz - 1) for x, sz in zip(types[t], sizes)) for t in d]
        return (sum(x * x for x in dl), d)

    return sorted(dists, key=score)


def staged_search(sizes, types, dists, n, caps, look=True):
    pending = list(range(len(dists)))
    total = 0
    for cap in caps:
        nxt = []
        for i in pending:
            vt = [types[t] for t in dists[i]]
            s = Typed(sizes, vt, n, cap, look)
            r = s.run()
            total += s.nodes
            if s.aborted:
                nxt.append(i)
            elif r:
                return True, [], total
        pending = nxt
        if not pending:
            break
    return False, pending, total


def feasible_typed(parts, n, cap=None, look=True, filters=True, log=None):
    t0 = time.time()
    sizes, types, dists, raw = distributions(list(parts), n, filters)
    dists = order_distributions(sizes, types, dists, n)
    if log:
        log("    distributions: raw %d, after filters %d (%.1fs)" % (raw, len(dists), time.time() - t0))
    caps = [1500, 30000, cap]
    found, pending, total = staged_search(sizes, types, dists, n, caps, look)
    if found:
        return True, total
    if pending:
        return None, total
    return False, total


def run_dist(item):
    i, sizes, types, dist, n = item
    vt = [types[t] for t in dist]
    s = Typed(sizes, vt, n, None)
    r = s.run()
    return i, r, s.nodes


def typed_task(item):
    m, n, cap = item
    r, nodes = feasible_typed(list(m), n, cap)
    return m, r


def resolve(parts, n, pool, status, save, log):
    t0 = time.time()
    sizes, types, dists, raw = distributions(list(parts), n, True)
    dists = order_distributions(sizes, types, dists, n)
    key = "sub:" + ",".join(map(str, parts))
    done = set(status.get(key, []))
    log("  resolving %s: %d type distributions (%.1fs), %d already done" % (list(parts), len(dists), time.time() - t0, len(done)))
    todo = [(i, sizes, types, d, n) for i, d in enumerate(dists) if i not in done]
    found = False
    t1 = time.time()
    count = 0
    for i, r, nodes in pool.imap_unordered(run_dist, todo, chunksize=1):
        count += 1
        done.add(i)
        if r:
            found = True
            break
        if count % max(1, len(todo) // 20) == 0:
            status[key] = sorted(done)
            save()
            log("    %d/%d distributions, %.0fs" % (len(done), len(dists), time.time() - t1))
    status[key] = sorted(done)
    save()
    return found


def weight(m):
    return sum(k * (k - 1) // 2 for k in m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("n", type=int)
    ap.add_argument("--cap", type=int, default=300000)
    ap.add_argument("--runs", type=int, default=100000)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--state", default=None)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    n = args.n
    state_file = args.state or "state_%d.json" % n
    status = json.load(open(state_file)) if os.path.exists(state_file) else {}

    def log(msg):
        print(msg, flush=True)

    def save():
        json.dump(status, open(state_file, "w"))

    key = lambda m: ",".join(map(str, m))
    allm = multisets(n)
    log("n=%d candidates=%d" % (n, len(allm)))
    feas = random_phase(n, args.runs, args.seed)
    log("random phase: %d multisets" % len(feas))
    t0 = time.time()
    pool = Pool(args.workers)
    i = 0
    while i < len(allm):
        j = i
        w = weight(allm[i])
        while j < len(allm) and weight(allm[j]) == w:
            j += 1
        undecided = []
        for m in allm[i:j]:
            k = key(m)
            if k in status and status[k] is not None:
                continue
            if any(status.get(key(d)) is False for d in lower_neighbors(m)):
                status[k] = False
                continue
            if len(m) > packbound(n) or not outside_ok(m, n):
                status[k] = False
                continue
            if dominated(m, feas):
                status[k] = True
                continue
            undecided.append(m)
        if undecided:
            hard = []
            for m, r in pool.imap_unordered(typed_task, [(m, n, args.cap) for m in undecided], chunksize=1):
                if r is None:
                    hard.append(m)
                else:
                    status[key(m)] = r
            for m in hard:
                log("  heavy multiset %s" % (list(m),))
                status[key(m)] = resolve(m, n, pool, status, save, log)
            save()
            log("  weight %d: %d solved, %d heavy, elapsed %.0fs" % (w, len(undecided), len(hard), time.time() - t0))
        i = j
    save()
    pool.close()
    count = sum(1 for k, v in status.items() if v is True and not k.startswith("sub:"))
    log("n=%d count=%d time=%.0fs" % (n, count, time.time() - t0))


if __name__ == "__main__":
    main()
