# replay_cache.py <trace> [resident layers to drop from the host set ...]: replay a routing trace through a model of the expert cache and
# print the hit rate per layer.  Model: one pool a card (1600 slots for layers < 40, 1664 for the others), a decayed frequency per
# expert (half-life 64 steps), a miss is admitted when its frequency beats the lowest of 64 sampled cached experts (or a slot is free).
import json, random, struct, sys
HALF, SAMPLES = 64.0, 64
def records(path):
    data = open(path, "rb").read(); assert data[:4] == b"MCT1"; pos = 4
    def take(fmt):
        nonlocal pos
        v = struct.unpack_from("<" + fmt, data, pos); pos += struct.calcsize("<" + fmt); return v
    _, _, n_group = take("3I")
    for _ in range(n_group):
        _, _, nl = take("3I"); take(f"{nl}I"); take("2d"); take("3Q"); (nt,) = take("I"); take(f"{nt}Q")
    while pos + 12 <= len(data):
        il, n_used, n = struct.unpack_from("<3i", data, pos); pos += 12
        if pos + 4 * n > len(data): break
        yield il, struct.unpack_from(f"<{n}i", data, pos); pos += 4 * n
def simulate(path, drop=(), slots=(1600, 1664), seed=1):
    rng = random.Random(seed)
    score, last = {}, {}                      # (layer, expert) -> decayed frequency, step of its last update
    cached = [[], []]; where = [{}, {}]       # a card: the cached keys, and each key's index
    hit, tot, fills = {}, {}, 0
    step, first = 0, None
    def cur(k): return score[k] * 0.5 ** ((step - last[k]) / HALF)
    for il, ids in records(path):
        if il in drop: continue
        if first is None: first = il
        if il == first: step += 1
        c = 0 if il < 40 else 1
        for e in set(ids):
            k = (il, e)
            s = cur(k) + 1.0 if k in score else 1.0
            score[k] = s; last[k] = step
            tot[il] = tot.get(il, 0) + 1
            if k in where[c]:
                hit[il] = hit.get(il, 0) + 1; continue
            if len(cached[c]) < slots[c]:
                where[c][k] = len(cached[c]); cached[c].append(k); fills += 1; continue
            v = min(rng.sample(cached[c], SAMPLES), key=cur)
            if s > cur(v):
                i = where[c].pop(v); cached[c][i] = k; where[c][k] = i; fills += 1
    return hit, tot, fills, step
if __name__ == "__main__":
    drop = set(map(int, sys.argv[2:]))
    hit, tot, fills, steps = simulate(sys.argv[1], drop)
    H, T = sum(hit.values()), sum(tot.values())
    print(f"{sys.argv[1]}: {steps} steps, {len(tot)} layers, hit rate {100*H/T:.1f} % ({T/steps:.0f} unique selections a step, {fills/steps:.1f} fills a step)")
    for c, name in ((0, "card 0"), (1, "card 1")):
        ls = [l for l in tot if (l < 40) == (c == 0)]
        print(f"  {name}: {len(ls)} host layers, hit rate {100*sum(hit.get(l,0) for l in ls)/sum(tot[l] for l in ls):.1f} %")
    print("  per layer: " + " ".join(f"{l}:{100*hit.get(l,0)/tot[l]:.0f}" for l in sorted(tot)))
    json.dump({str(l): hit.get(l, 0) / tot[l] for l in tot}, open(sys.argv[1] + ".sim.json", "w"))
