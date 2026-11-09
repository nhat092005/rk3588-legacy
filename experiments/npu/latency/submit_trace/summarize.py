import re, sys, collections
def parse(path):
    lines = open(path).read().splitlines(); subs = []; cur = None
    for i, l in enumerate(lines):
        if l.startswith("SUBMIT"):
            m = re.search(r"n=(\d+) core_mask=(0x[0-9a-f]+)", l)
            cur = {"n": int(m[1]), "mask": m[2], "tasks": collections.defaultdict(list)}; subs.append(cur)
        elif l.startswith("TASK"):
            m = re.search(r"sub=(\d) idx=(\d+) op=(\d+) .* en=(0x[0-9a-f]+) .* amount=(\d+)", l)
            cur["tasks"][int(m[1])].append((m[4], [int(w, 16) for w in lines[i + 1].split()[1:]]))
    return subs
def reg(cmd, tg, r):
    for w in cmd:
        if (w >> 48) == tg and (w & 0xffff) == r: return (w >> 16) & 0xffffffff
KIND = {"0x18": "elementwise", "0x60": "pool", "0x1d": "conv", "0xd": "conv_nordma"}
for path in sys.argv[1:]:
    subs = parse(path); per = max(s["n"] for s in subs) + 1
    half = per // 2  # two inferences in the trace
    kinds = collections.Counter(); trip = collections.Counter(); conv1 = 0; allmask = collections.Counter()
    for s in subs:
        if s["n"] >= half: continue
        for t in s["tasks"].get(0 if s["mask"] == "0x1" else 2, []):
            allmask[(s["mask"], KIND.get(t[0], t[0]))] += 1
        if s["mask"] != "0x7": continue
        for t0, t1, t2 in zip(*[s["tasks"][k] for k in (2, 3, 4)]):
            hs = [reg(t[1], 0x1001, 0x4034) for t in (t0, t1, t2)]
            if None in hs: continue
            trip[tuple(h + 1 for h in hs)] += 1
    print(path)
    print("  submits per inference:", half, " tasks by (mask, kind) for first inference:", dict(allmask))
    print("  height triplets (rows per core, count):", dict(trip))
