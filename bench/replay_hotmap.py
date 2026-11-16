# replay_hotmap.py <trace> [K ...]: what per-expert placement would give, replayed on a routing trace (host layers of that run only).
# Layout "hot K": the K hottest experts of every layer live on the cards only (chosen from the FIRST half of the run), and the
# other experts share a cache.  The GPU memory is what the cards have today, scaled to the trace's layers: 24 resident layers
# out of 71 selectable ones and 3,264 cache slots become, for L traced layers, L*24/71*256 exclusive experts and L*3264/71 slots.
# Scored on the SECOND half of the run.  K = 0 is the cache alone with that many slots plus the whole-layer residency of today.
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from replay_cache import records, HALF, SAMPLES
import random
def run(path, K, n_slots_total, hot=None, seed=1):
    rng = random.Random(seed); recs = list(records(path)); first = recs[0][0]
    steps_total = sum(1 for il, _ in recs if il == first); half = steps_total // 2
    layers = sorted({il for il, _ in recs})
    if hot is None:
        cnt = {}; step = 0
        for il, ids in recs:
            if il == first: step += 1
            if step > half: break
            c = cnt.setdefault(il, {})
            for e in ids: c[e] = c.get(e, 0) + 1
        hot = {il: set(sorted(cnt.get(il, {}), key=lambda e: -cnt[il][e])[:K]) for il in layers}
    n0 = sum(1 for l in layers if l < 40)
    slots = (round(n_slots_total * n0 / len(layers)), round(n_slots_total * (len(layers) - n0) / len(layers)))
    score, last = {}, {}; cached = [[], []]; where = [{}, {}]
    step = 0; H = S = T = 0
    def cur(k): return score[k] * 0.5 ** ((step - last[k]) / HALF)
    for il, ids in recs:
        if il == first: step += 1
        c = 0 if il < 40 else 1; hs = hot[il]
        for e in set(ids):
            scored = step > half
            if scored: T += 1
            if e in hs:
                if scored: S += 1
                continue
            k = (il, e); s = cur(k) + 1.0 if k in score else 1.0
            score[k] = s; last[k] = step
            if k in where[c]:
                if scored: H += 1
                continue
            if len(cached[c]) < slots[c]:
                where[c][k] = len(cached[c]); cached[c].append(k); continue
            if slots[c] == 0: continue
            v = min(rng.sample(cached[c], min(SAMPLES, len(cached[c]))), key=cur)
            if s > cur(v):
                i = where[c].pop(v); cached[c][i] = k; where[c][k] = i
    return len(layers), S / T, H / T
if __name__ == "__main__":
    path = sys.argv[1]; Ks = list(map(int, sys.argv[2:])) or [0, 87, 110, 130]
    for K in Ks:
        L = len({il for il, _ in records(path)})
        excl_today = L * 24 / 71 * 256          # exclusive experts the cards hold today, for these L layers
        slots_today = L * 3264 / 71
        if K == 0:
            n, s, h = run(path, 0, L * 3264 / 47)   # today the 3,264 slots serve the 47 host layers alone
            gpu = (24 / 71) + (47 / 71) * h      # today: whole layers resident, the cache on the others
            print(f"today (whole layers + cache): cache hit {100*h:.1f} % on host layers -> GPU share of the 71 layers {100*gpu:.1f} %")
        else:
            extra = excl_today - K * L             # the same GPU memory: more hot experts leave fewer cache slots, and the reverse
            ns = max(slots_today + extra, 0)
            n, s, h = run(path, K, ns)
            print(f"hot {K:3d} a layer ({100*K/256:.0f} % of its experts, exclusive) + cache of {ns:.0f} slots{' [host RAM exceeded]' if K < 87 else ''}: "
                  f"static {100*s:.1f} % + cache {100*h:.1f} % = GPU share {100*(s+h):.1f} %   (host holds {256-K} experts a layer)")
