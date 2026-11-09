import re, sys, collections
# parse trace: for each 3-core submit, compare per-core task register commands
lines = open(sys.argv[1]).read().splitlines()
subs = []  # (n, mask, sublist, tasks{sub:[(idx,op,en,amount,cmd[])]})
cur = None
for i, l in enumerate(lines):
    if l.startswith("SUBMIT"):
        m = re.search(r"n=(\d+) core_mask=(0x[0-9a-f]+)", l)
        sub = [tuple(map(int, x.split(","))) for x in re.findall(r"\[(\d+,\d+)\]", l)]
        cur = {"n": int(m[1]), "mask": m[2], "sub": sub, "tasks": collections.defaultdict(list)}
        subs.append(cur)
    elif l.startswith("TASK"):
        m = re.search(r"sub=(\d) idx=(\d+) op=(\d+) .* en=(0x[0-9a-f]+) .* amount=(\d+)", l)
        cmd = lines[i + 1].split()[1:]
        cur["tasks"][int(m[1])].append((int(m[2]), int(m[3]), m[4], int(m[5]), [int(w, 16) for w in cmd]))
def dec(w): return (w >> 48, (w >> 16) & 0xffffffff, w & 0xffff)
rows = collections.Counter()
for s in subs:
    if s["mask"] != "0x7": continue
    c = [s["tasks"][k] for k in (2, 3, 4)]
    for t0, t1, t2 in zip(*c):
        diffregs = []
        for w0, w1, w2 in zip(t0[4], t1[4], t2[4]):
            if w0 != w1 or w0 != w2:
                tg, v0, r = dec(w0); v1 = dec(w1)[1]; v2 = dec(w2)[1]
                diffregs.append((hex(tg), hex(r), v0, v1, v2))
        print("submit", s["n"], "task", t0[0], "op", t0[1], "en", t0[2], "ndiff", len(diffregs))
        for d in diffregs: print("   ", d)
