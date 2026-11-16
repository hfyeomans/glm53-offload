# equiv.py <out.json>: asks the server at 127.0.0.1:8080 five prompts (16 to about 9,000 tokens), greedy, 120 tokens, each twice,
# and saves every generated token with its top-5 probabilities.  For comparing two engines or presets on the same model.
# equiv.py --cmp a.json b.json: per prompt, how many leading tokens are identical, the largest difference in the chosen token's
#   probability over them, and the top-3 of both at the first token that differs.  With a.json twice: run 1 against run 2.
# equiv.py --first a.json b.json: the top-3 of the first generated token (the prompt path alone), both runs of both files.
# The engine does not repeat itself exactly, so compare the difference between two engines with the spread inside one.
import json, math, os, sys, urllib.request

def ask(prompt, n):
    body = {"prompt": prompt, "n_predict": n, "temperature": 0, "seed": 1, "n_probs": 5, "cache_prompt": False}
    r = urllib.request.Request("http://127.0.0.1:8080/completion", json.dumps(body).encode(), {"Content-Type": "application/json"})
    d = json.load(urllib.request.urlopen(r, timeout=1200))
    toks = [(t.get("token", ""), t.get("logprob"), [(x.get("token", ""), x.get("logprob")) for x in t.get("top_logprobs") or []])
            for t in d.get("completion_probabilities", [])]
    return {"content": d["content"], "tokens": toks, "n_prompt": d.get("tokens_evaluated")}

def P(lp): return math.exp(lp) if lp is not None else float("nan")
def top3(t): return ", ".join(f"{tok!r} {P(lp):.4f}" for tok, lp in t[2][:3])

def cmp(name, ta, tb):
    n = 0; md = 0.0
    while n < min(len(ta), len(tb)) and ta[n][0] == tb[n][0]:
        md = max(md, abs(P(ta[n][1]) - P(tb[n][1]))); n += 1
    print(f"{name}: {n} of {len(ta)}/{len(tb)} generated tokens identical, largest probability difference over them {md:.4f}")
    if n < min(len(ta), len(tb)):
        print("  a top:", top3(ta[n])); print("  b top:", top3(tb[n]))

if __name__ == "__main__":
    if sys.argv[1] in ("--cmp", "--first"):
        a, b = (json.load(open(p)) for p in sys.argv[2:4])
        for k in sorted({k[:-2] for k in a}):
            if sys.argv[1] == "--first":
                print(k)
                for lab, d in (("a", a), ("b", b)):
                    for rep in ("#1", "#2"): print(f"  {lab}{rep}:", top3(d[k + rep]["tokens"][0]))
            elif sys.argv[2] == sys.argv[3]:
                cmp(k + " run 1 vs run 2", a[k + "#1"]["tokens"], a[k + "#2"]["tokens"])
            else:
                for rep in ("#1", "#2"): cmp(k + rep, a[k + rep]["tokens"], b[k + rep]["tokens"])
    else:
        long = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "equiv-long.txt")).read()
        prompts = {
            "short": "The three laws of thermodynamics, stated plainly, are as follows.\n1.",
            "list": "Here is a numbered list of the twelve months of the year with the number of days in each:\n1. January - 31\n2.",
            "code": "def quicksort(xs):\n    \"\"\"Sort a list with quicksort and return a new list.\"\"\"\n",
            "mid": long[:3500] + "\n\nIn one paragraph, the text above says that",
            "long": long + "\n\nIn one paragraph, the text above says that",
        }
        out = {}
        for rep in (1, 2):
            for k, p in prompts.items():
                r = ask(p, 120); out[f"{k}#{rep}"] = r
                print(k, rep, r["n_prompt"], len(r["tokens"]), repr(r["content"][:70]), flush=True)
        json.dump(out, open(sys.argv[1], "w"))
