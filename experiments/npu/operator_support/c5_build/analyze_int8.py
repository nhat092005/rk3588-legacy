import re,collections,sys
S="/tmp/claude-1000/-home-nhat-workspace-prjs-rk3588/d98a5996-660b-48b1-8f49-5240c06786b7/scratchpad"
def table(path):
    L=open(path,encoding="utf8",errors="replace").read().replace("\r","\n").split("\n")
    h=next(i for i,l in enumerate(L) if "OpType" in l and "Target" in l)
    rows=[]
    for l in L[h+1:]:
        m=re.search(r"\] (\d+)\s+(\w+)\s+(\w+)\s+(NPU|CPU)\s+\S+\s+\S+\s+(\d+)/(\d+)/(\d+)\s+(\d+)\s+(\S+)",l)
        if m: rows.append(dict(id=int(m.group(1)),op=m.group(2),dt=m.group(3),tgt=m.group(4),tot=int(m.group(7))))
        elif rows: break
    return rows
for m in ("baseline","relu"):
    r=table(f"{S}/c5_{m}/int8_stdout.txt")
    tot=sum(x["tot"] for x in r)
    print(f"== {m} INT8: rows {len(r)} total cycles {tot}")
    by=collections.defaultdict(lambda:[0,0,collections.Counter(),collections.Counter()])
    for x in r:
        b=by[x["op"]]; b[0]+=1; b[1]+=x["tot"]; b[2][x["tgt"]]+=1; b[3][x["dt"]]+=1
    for op,(n,c,t,d) in sorted(by.items(),key=lambda kv:-kv[1][1]):
        print(f"  {op:14s} n={n:3d} cycles={c:9d} share={100*c/tot:5.2f}% tgt={dict(t)} dtype={dict(d)}")
    print("  CPU rows:",[(x['id'],x['op'],x['dt']) for x in r if x['tgt']=='CPU'])
