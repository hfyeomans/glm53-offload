# switch.py <out.json> <step>...: conversations switched on one slot.  A step is NAME:TOKENS (a conversation of about that many
# tokens, the same text every time it is named) or "side" (a short unrelated request).  Prints, per step, the tokens the server
# computed and reused, the prompt time and the wall time to the end of a 16-token answer.
import json, random, sys, time, urllib.request
URL = "http://127.0.0.1:8080/v1/chat/completions"
WORDS = "the of and to in is that it for as with was on be at by this had not are but from or have an they which one you were her all she there would their we him been has when who will more no if out so said what up its about into than them can only other new some could time these two may then do first any my now such like our over man me even most made after also did many before must through back years where much your way well down should because each just those people how too little state good very make world still own see men work long get here between both life being under never day same another know while last might us great old year off come since against go came right used take three".split()
def conv(name, n):
    r = random.Random(name)
    return "[conversation %s]\n" % name + " ".join(r.choice(WORDS) for _ in range(n)) + "\n\nIn one sentence, what is the text above?"
def ask(text):
    body = json.dumps({"model": "glm-5.3", "messages": [{"role": "user", "content": text}], "max_tokens": 16, "temperature": 0}).encode()
    t0 = time.time()
    with urllib.request.urlopen(urllib.request.Request(URL, body, {"Content-Type": "application/json"}), timeout=3600) as f:
        j = json.load(f)
    m = j["choices"][0]["message"]
    return time.time() - t0, dict(j.get("timings", {}), text=(m.get("reasoning_content") or "") + "|" + (m.get("content") or ""))
out = []
for i, step in enumerate(sys.argv[2:]):
    if step == "side":
        name, text = "side", "Say hi, request %d." % i
    else:
        name, n = step.split(":"); text = conv(name, int(n))
    wall, t = ask(text)
    rec = {"step": step, "wall_s": round(wall, 2), "prompt_n": t.get("prompt_n"), "cache_n": t.get("cache_n"), "prompt_ms": t.get("prompt_ms"), "predicted_ms": t.get("predicted_ms"), "text": t.get("text")}
    rec["before_first_token_s"] = round(wall - (t.get("predicted_ms") or 0) / 1000, 2)
    out.append(rec); print(rec, flush=True)
    json.dump(out, open(sys.argv[1], "w"), indent=1)
