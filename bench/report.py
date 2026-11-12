#!/usr/bin/env python3
"""report.py -- read the files scripts/bench.sh writes.

  report.py bench.json [more.json ...]
      print each bench: decode over the prompts, then every prompt step.

  report.py --tables [--repo DIR] [--draft]
      fill the tables of README.md and RESULTS.md (the blocks between <!-- T:name --> and <!-- /T:name -->) from
        results/<preset>.json           scripts/bench.sh against the preset as it ships (route bias 0)
        results/<preset>-bias003.json   scripts/bench.sh --decode-only against the preset started with --route-bias 0.03
        recipes/<preset>.sh             the context, the cache size and the host memory
      for the ten presets (1gpu-64k ... 2gpu-1m).  It fails if a file is missing or a block is left unfilled; --draft writes
      what it can.

Python 3, stdlib only.
"""
import argparse, json, os, re, statistics, sys

RUNGS = ("64k", "128k", "256k", "512k", "1m")
PRESETS = [f"{n}gpu-{r}" for n in (1, 2) for r in RUNGS]
DASH = "–"


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def decode(j):
    """decode tok/s over the prompts (the figure bench.sh prints: the median of the prompts' medians) and the time to first token"""
    runs = [r for v in j.get("decode", {}).values() for r in v.get("runs", [])]
    d = [v["median_decode_tps"] for v in j.get("decode", {}).values() if v.get("median_decode_tps")]
    t = [v["median_ttft_s"] for v in j.get("decode", {}).values() if v.get("median_ttft_s")]
    if not d:
        return None
    return {"n": len(d), "med": statistics.median(d), "lo": min(d), "hi": max(d), "ttft": statistics.median(t) if t else None,
            "tokens": max((r.get("completion_tokens") or 0) for r in runs) if runs else None,
            "short": sum(1 for r in runs if r.get("finish") != "length")}


def f1(x):
    return DASH if x is None else f"{x:.1f}"


def f2(x):
    return DASH if x is None else f"{x:.2f}"


def f0(x):
    return DASH if x is None else f"{round(x):,}"


def ktok(n):
    """3,610 -> 3.6K; 25,230 -> 25K; 1,000,000 -> 1M"""
    if n is None:
        return DASH
    if n < 10000:
        return f"{n / 1000:.1f}K"
    if n < 999500:
        return f"{round(n / 1000)}K"
    return f"{n / 1e6:.2f}".rstrip("0").rstrip(".") + "M"


def summary(path):
    j = load(path)
    print(f"{path}: {j.get('model')} at {j.get('url')}, started {j.get('started')}")
    d = decode(j)
    if d:
        print(f"  decode  {d['n']} prompts, up to {d['tokens']} tokens each: median {d['med']:.1f} tok/s "
              f"(lowest prompt {d['lo']:.1f}, highest {d['hi']:.1f}), first token after {f2(d['ttft'])} s (median)")
        if d["short"]:
            print(f"          {d['short']} of the requests ended before their token limit")
    else:
        print("  decode  no figures in this file")
    for p in j.get("prefill", []):
        print(f"  prompt  grown to {p['prompt_tokens']:,} tokens: {f0(p.get('prompt_tps'))} tok/s, first token after {f1(p.get('ttft_s'))} s; "
              f"then {p.get('after_tokens')} tokens at {f1(p.get('decode_tps_after'))} tok/s")


def recipe(repo, preset):
    out = {}
    with open(os.path.join(repo, "recipes", f"{preset}.sh"), encoding="utf-8") as f:
        for line in f:
            m = re.match(r"(CTX|CACHE_MIB|NEED_RAM_GIB)=([0-9]+)", line)
            if m:
                out[m.group(1)] = int(m.group(2))
    return out


def fill(text, name, body, where):
    pat = re.compile(r"(<!-- T:%s -->\n).*?(<!-- /T:%s -->)" % (name, name), re.S)
    if not pat.search(text):
        raise SystemExit(f"report.py: {where} has no block T:{name}")
    return pat.sub(lambda m: m.group(1) + body.rstrip("\n") + "\n" + m.group(2), text, count=1)


def tables(repo, draft):
    missing = []

    def bench(name):
        p = os.path.join(repo, "results", f"{name}.json")
        if not os.path.exists(p):
            missing.append(f"results/{name}.json")
            return None
        return load(p)

    R = {p: {"b0": bench(p), "b3": bench(f"{p}-bias003"), "rec": recipe(repo, p)} for p in PRESETS}
    cards = lambda p: p[0]
    ctx = lambda p: p.split("-")[1].upper()

    def dec(j):
        d = decode(j) if j else None
        return d["med"] if d else None

    def steps(j):
        return (j or {}).get("prefill", [])

    # README: the 64K presets
    head = steps(R["1gpu-64k"]["b0"]) or steps(R["2gpu-64k"]["b0"])
    cols = [ktok(s["prompt_tokens"]) for s in head] or [DASH] * 3
    rows = ["| 64K context | Decode | " + " | ".join(("Prompt, to " if i == 0 else "to ") + c + (" tokens" if i == 0 else "") for i, c in enumerate(cols)) + " |",
            "|---|" + "---|" * (1 + len(cols))]
    for p, label in (("1gpu-64k", "One card"), ("2gpu-64k", "Two cards")):
        st = steps(R[p]["b0"])
        cells = [f0(s.get("prompt_tps")) + (" tok/s" if i == 0 else "") for i, s in enumerate(st)] or [DASH] * len(cols)
        rows.append(f"| **{label}** | **{f1(dec(R[p]['b0']))} tok/s** | " + " | ".join(cells) + " |")
    headline = "\n".join(rows)

    # README: every preset
    rows = ["| Cards | Context | Decode, tok/s | With `--route-bias 0.03` | Prompt, tok/s (deepest step) | Decode at that depth, tok/s | Host RAM in use |",
            "|---|---|---|---|---|---|---|"]
    for p in PRESETS:
        a, b, st = dec(R[p]["b0"]), dec(R[p]["b3"]), steps(R[p]["b0"])
        gain = f" ({(b / a - 1) * 100:+.0f} %)" if a and b else ""
        deep = st[-1] if st else None
        rows.append(f"| {cards(p)} | {ctx(p)} | {f1(a)} | {f1(b)}{gain} | "
                    + (f"{f0(deep.get('prompt_tps'))} (to {ktok(deep['prompt_tokens'])})" if deep else DASH)
                    + f" | {f1(deep.get('decode_tps_after')) if deep else DASH} | {R[p]['rec'].get('NEED_RAM_GIB', DASH)} GiB |")
    presets = "\n".join(rows)

    # RESULTS: decode
    rows = ["| Preset | Decode, tok/s | Lowest prompt | Highest prompt | First token, s | With `--route-bias 0.03` | Gain |",
            "|---|---|---|---|---|---|---|"]
    for p in PRESETS:
        d0 = decode(R[p]["b0"]) if R[p]["b0"] else None
        d3 = decode(R[p]["b3"]) if R[p]["b3"] else None
        gain = f"{(d3['med'] / d0['med'] - 1) * 100:+.1f} %" if d0 and d3 else DASH
        rows.append(f"| `{p}` | {f1(d0 and d0['med'])} | {f1(d0 and d0['lo'])} | {f1(d0 and d0['hi'])} | "
                    f"{f2(d0 and d0['ttft'])} | {f1(d3 and d3['med'])} | {gain} |")
    t_decode = "\n".join(rows)

    # RESULTS: prompt steps
    rows = ["| Preset | Prompt grown to | Prompt, tok/s | First token after, s | Decode after it, tok/s |", "|---|---|---|---|---|"]
    for p in PRESETS:
        st = steps(R[p]["b0"])
        if not st:
            rows.append(f"| `{p}` | {DASH} | {DASH} | {DASH} | {DASH} |")
        for i, s in enumerate(st):
            rows.append(f"| {'`' + p + '`' if i == 0 else ''} | {s['prompt_tokens']:,} | {f0(s.get('prompt_tps'))} | {f1(s.get('ttft_s'))} | {f1(s.get('decode_tps_after'))} |")
    t_prompt = "\n".join(rows)

    # RESULTS: what each preset is sized for
    rows = ["| Preset | Context, tokens | Expert cache, MiB | Host RAM in use at the peak, GiB |", "|---|---|---|---|"]
    for p in PRESETS:
        r = R[p]["rec"]
        rows.append(f"| `{p}` | {r.get('CTX', 0):,} | {r.get('CACHE_MIB', 0):,} | {r.get('NEED_RAM_GIB', DASH)} |")
    t_sizes = "\n".join(rows)

    left = []
    for name, blocks in (("README.md", {"headline": headline, "presets": presets}),
                         ("RESULTS.md", {"decode": t_decode, "prompt": t_prompt, "sizes": t_sizes})):
        path = os.path.join(repo, name)
        with open(path, encoding="utf-8") as f:
            text = f.read()
        for k, body in blocks.items():
            text = fill(text, k, body, name)
        left += [f"{name}: T:{m}" for m in re.findall(r"<!-- T:([a-z0-9-]+) -->\n(?=[^\n]*pending)", text)]
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
        print(f"{name}: {len(blocks)} tables written")
    for m in missing:
        print(f"report.py: missing {m}", file=sys.stderr)
    for m in left:
        print(f"report.py: still pending, {m}", file=sys.stderr)
    if (missing or left) and not draft:
        raise SystemExit(1)


def main():
    ap = argparse.ArgumentParser(description="Read the files scripts/bench.sh writes.")
    ap.add_argument("files", nargs="*", help="bench JSON files to print")
    ap.add_argument("--tables", action="store_true", help="fill the tables of README.md and RESULTS.md from results/")
    ap.add_argument("--repo", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."), help="the repo (default: this one)")
    ap.add_argument("--draft", action="store_true", help="with --tables: write what there is, do not fail on what is missing")
    a = ap.parse_args()
    if a.tables:
        tables(a.repo, a.draft)
    elif a.files:
        for p in a.files:
            summary(p)
    else:
        ap.print_help()
        raise SystemExit(2)


if __name__ == "__main__":
    main()
