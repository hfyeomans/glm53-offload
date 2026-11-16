# hotmap.py --pin N0,N1 <trace> [<trace> ...] > map: the hot-expert map for LLAMA_MOE_HOT_MAP, from routing traces
# (LLAMA_MOE_CACHE_TRACE).  A card pins the N busiest experts of its own layers (layers below SPLIT are on card 0): they live on the
# card only, and host memory holds the others.  A layer that several traces cover takes the mean of its rates.
#   --pin 3584,2560   what 14 and 10 whole layers free on the cards of the 256K preset (256 experts a layer)
#   --max 200         the most experts one layer may pin
#   --part first      pick from the first half of every trace only (or second; default all), to test the map on prompts it has not seen
#   --holdout         pick from the first half of every trace, print the share of the second half's traffic the pins take, write no map
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from replay_cache import records

SPLIT, N_EXPERT = 40, 256

def rates(paths, part):
    """(layer, expert) -> selections a step, a step counting an expert once; part: 'all', 'first' or 'second' half of each trace"""
    acc = {}
    for path in paths:
        recs = list(records(path)); first = recs[0][0]
        steps = sum(1 for il, _ in recs if il == first); half = steps // 2
        lo, hi = {"all": (1, steps), "first": (1, half), "second": (half + 1, steps)}[part]
        cnt, step = {}, 0
        for il, ids in recs:
            if il == first: step += 1
            if step < lo or step > hi: continue
            for e in set(ids):
                cnt[(il, e)] = cnt.get((il, e), 0) + 1
        n = hi - lo + 1
        for l in {il for il, _ in cnt}:
            for e in range(N_EXPERT):
                acc.setdefault((l, e), []).append(cnt.get((l, e), 0) / n)
    return {k: sum(v) / len(v) for k, v in acc.items()}

def pick(rate, pins, cap):
    hot = {}
    for card, n in enumerate(pins):
        keys = sorted((k for k in rate if (k[0] < SPLIT) == (card == 0)), key=lambda k: -rate[k])
        for l, e in keys:
            if n == 0: break
            if len(hot.setdefault(l, [])) < cap:
                hot[l].append(e); n -= 1
        if n: sys.exit(f"card {card}: {n} pins left over, raise --max")
    return hot

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pin", required=True); ap.add_argument("--max", type=int, default=200)
    ap.add_argument("--part", default="all", choices=("all", "first", "second"))
    ap.add_argument("--holdout", action="store_true"); ap.add_argument("traces", nargs="+")
    a = ap.parse_args()
    pins = [int(x) for x in a.pin.split(",")]
    if a.holdout:
        hot = pick(rates(a.traces, "first"), pins, a.max); test = rates(a.traces, "second")
        for card in (0, 1):
            ks = [k for k in test if (k[0] < SPLIT) == (card == 0)]
            tot = sum(test[k] for k in ks); got = sum(test[k] for k in ks if k[1] in hot.get(k[0], ()))
            ns = [len(hot.get(l, ())) for l in sorted({k[0] for k in ks})]
            print(f"card {card}: {len(ns)} layers, {sum(ns)} pins ({min(ns)} to {max(ns)} a layer), {100*got/tot:.1f} % of the second half's traffic")
        tot = sum(test.values()); got = sum(v for k, v in test.items() if k[1] in hot.get(k[0], ()))
        print(f"both cards: {100*got/tot:.1f} % of the traffic on pinned experts")
    else:
        rate = rates(a.traces, a.part); hot = pick(rate, pins, a.max)
        print(f"# hot-expert map: layer, then the experts the card pins, busiest first.  {sum(map(len, hot.values()))} experts of {len(hot)} layers,")
        print(f"# --pin {a.pin} --max {a.max} --part {a.part}, from {' '.join(os.path.basename(p) for p in a.traces)}")
        for l in sorted(hot):
            print(l, *hot[l])
