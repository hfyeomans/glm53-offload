#!/usr/bin/env python3
"""probe.py -- speed probe for an OpenAI-compatible llama-server offload cell (tools/offload/run-cell.sh).

Decode: one warmup, then N x 512-token story and code answers, greedy, enable_thinking=false sent (a template without that switch -- GLM-5.3 -- thinks anyway: render it with /apply-template); per request it records
  ttft_s        client: seconds to the first non-empty content OR reasoning_content delta
  decode_tps    client: (completion_tokens - 1) / (last delta - first delta)   -- the decode phase only
  engine_tps    server timings.predicted_per_second, prompt_tps = timings.prompt_per_second (llama-server)
  draft stats   timings.draft_n / draft_n_accepted when a drafter is on
Prefill: one request per prompt length (4K, 32K, 64K by default): prompt tok/s from the server's timings, TTFT client-side.
Token counts come from usage/timings, never from counting deltas (a drafter emits several tokens per delta). Stdlib only.

    python3 probe.py --url http://127.0.0.1:18090 --model <alias> --out probe.json [--reps 5] [--max-tokens 512]
                     [--prefill 4096,32768,65536] [--no-prefill] [--think]
"""
import argparse, json, statistics, time, urllib.request

PROMPTS = {
    "story": "Write a long, detailed story (at least 1,500 words) about a lighthouse keeper who finds a message in a "
             "bottle that was written by her own grandmother. Use dialogue and vivid description.",
    "code": "Write a complete Python module implementing a red-black tree: insert, delete, search, in-order traversal, "
            "and a validate() method that checks every red-black invariant. Include docstrings and type hints. "
            "Output only the code.",
}
FILLER = ("The quick brown fox jumps over the lazy dog while the committee reviews the quarterly figures and the "
          "engineers recalibrate the sensors on the northern array. ")   # ~32 tokens per repeat (approximate)
TIMEOUT = 7200   # seconds per request; --timeout (a 500K-token prompt segment at depth can run past 2 h)


def chat(base, model, messages, max_tokens, think):
    body = {"model": model, "stream": True, "max_tokens": max_tokens, "temperature": 0, "messages": messages,
            "stream_options": {"include_usage": True}, "timings_per_token": False}
    body["chat_template_kwargs"] = {"enable_thinking": bool(think)}
    req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter(); first = last = None; usage = None; timings = None; finish = None; n_deltas = 0; text = []; stamps = []
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            obj = json.loads(payload)
            if obj.get("usage"):
                usage = obj["usage"]
            if obj.get("timings"):
                timings = obj["timings"]
            for ch in obj.get("choices", []):
                d = ch.get("delta", {})
                piece = d.get("content") or d.get("reasoning_content") or ""
                if piece:
                    now = time.perf_counter()
                    if first is None:
                        first = now
                    last = now; n_deltas += 1; text.append(piece); stamps.append(now)
                if ch.get("finish_reason"):
                    finish = ch["finish_reason"]
    t1 = time.perf_counter()
    ct = (usage or {}).get("completion_tokens") or (timings or {}).get("predicted_n") or 0
    pt = (usage or {}).get("prompt_tokens") or (timings or {}).get("prompt_n") or 0
    dec = (ct - 1) / (last - first) if (first and last and last > first and ct > 1) else None
    return {"ttft_s": (first - t0) if first else None, "decode_tps": dec, "completion_tokens": ct, "prompt_tokens": pt,
            "finish": finish, "wall_s": t1 - t0, "n_deltas": n_deltas, "timings": timings,
            "text_head": "".join(text)[:200], "stamps": stamps}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True); ap.add_argument("--model", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=5); ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--prefill", default="4096,32768,65536"); ap.add_argument("--no-prefill", action="store_true")
    ap.add_argument("--after-tokens", type=int, default=32)   # tokens generated after each prefill prompt; >= 128 gives a decode_curve_64
    ap.add_argument("--after-prompt", default="In one sentence, what is the text above about?")   # the question after the filler text
    ap.add_argument("--after-prompt-file", default=None)   # read the question from a file (the runner word-splits PROBE_ARGS)
    ap.add_argument("--prefill-file", default=None)   # a real document for the prefill prompts instead of the repeated filler (cut to the target)
    ap.add_argument("--think", action="store_true")
    ap.add_argument("--salt", action="store_true")   # a unique first line on each prefill prompt: a repeated size is a fresh prompt, not a prefix-cache hit
    ap.add_argument("--timeout", type=int, default=7200)
    a = ap.parse_args()
    global TIMEOUT; TIMEOUT = a.timeout
    out = {"url": a.url, "model": a.model, "think": a.think, "decode": {}, "prefill": [], "started": time.strftime("%F %T")}
    print("warmup", flush=True)
    w = chat(a.url, a.model, [{"role": "user", "content": PROMPTS["story"]}], 128, a.think)
    print(f"  warmup: {w['completion_tokens']} tok, ttft {w['ttft_s']:.2f}s, decode {w['decode_tps'] and round(w['decode_tps'],1)} tok/s", flush=True)
    for name, prompt in PROMPTS.items():
        runs = []
        for i in range(a.reps):
            r = chat(a.url, a.model, [{"role": "user", "content": prompt}], a.max_tokens, a.think)
            runs.append(r)
            tm = r["timings"] or {}
            print(f"  {name} rep {i+1}: {r['completion_tokens']} tok ttft {r['ttft_s']:.2f}s decode {r['decode_tps'] and round(r['decode_tps'],1)} "
                  f"engine {tm.get('predicted_per_second') and round(tm['predicted_per_second'],1)} draft {tm.get('draft_n')}/{tm.get('draft_n_accepted')} finish {r['finish']}", flush=True)
        dec = [r["decode_tps"] for r in runs if r["decode_tps"]]
        out["decode"][name] = {"runs": runs, "median_decode_tps": statistics.median(dec) if dec else None,
                               "median_ttft_s": statistics.median([r["ttft_s"] for r in runs if r["ttft_s"]])}
    if not a.no_prefill:
        for n in [int(x) for x in a.prefill.split(",") if x]:
            reps = max(1, n // 32)
            after = open(a.after_prompt_file, encoding="utf-8").read().strip() if a.after_prompt_file else a.after_prompt
            if a.prefill_file:
                doc = open(a.prefill_file, encoding="utf-8", errors="replace").read()
                body = doc[: int(n * 3.2)]   # ~3.2 chars per token for code; the server reports the real count
            else:
                body = FILLER * reps
            if a.salt:
                body = f"[prompt {len(out['prefill'])} {time.time_ns()}]\n" + body
            prompt = body + "\n\n" + after
            r = chat(a.url, a.model, [{"role": "user", "content": prompt}], a.after_tokens, a.think)
            tm = r["timings"] or {}
            rec = {"target_tokens": n, "prompt_tokens": r["prompt_tokens"], "ttft_s": r["ttft_s"], "prompt_tps": tm.get("prompt_per_second"),
                   "prompt_ms": tm.get("prompt_ms"), "decode_tps_after": r["decode_tps"], "after_tokens": r["completion_tokens"],
                   "draft_n_after": tm.get("draft_n"), "draft_n_accepted_after": tm.get("draft_n_accepted")}
            st = r["stamps"]   # per-delta timestamps: decode tok/s per successive 64-delta window = the hot-set recovery curve after a prompt
            rec["decode_curve_64"] = [round(64 / (st[k + 64] - st[k]), 1) for k in range(0, len(st) - 64, 64)]
            out["prefill"].append(rec)
            print(f"  prefill {r['prompt_tokens']} tok: ttft {r['ttft_s'] and round(r['ttft_s'],1)}s prompt {rec['prompt_tps'] and round(rec['prompt_tps'])} tok/s"
                  + (f" after {rec['after_tokens']} tok: decode {r['decode_tps'] and round(r['decode_tps'],1)} curve64 {rec['decode_curve_64']}" if rec["decode_curve_64"] else "")
                  + (f" draft {rec['draft_n_after']}/{rec['draft_n_accepted_after']}" if rec["draft_n_after"] is not None else ""), flush=True)
    s = out["decode"]
    print(f"SUMMARY {a.model}: story {s['story']['median_decode_tps'] and round(s['story']['median_decode_tps'],1)} / "
          f"code {s['code']['median_decode_tps'] and round(s['code']['median_decode_tps'],1)} tok/s decode (median of {a.reps} x {a.max_tokens}); "
          + " ".join(f"pp{p['prompt_tokens']}={p['prompt_tps'] and round(p['prompt_tps'])}" for p in out["prefill"]), flush=True)
    json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
