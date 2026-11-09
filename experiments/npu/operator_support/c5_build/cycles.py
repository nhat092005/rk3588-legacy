import re,collections,sys
S="/tmp/claude-1000/-home-nhat-workspace-prjs-rk3588/d98a5996-660b-48b1-8f49-5240c06786b7/scratchpad"
def table(path):
    L=open(path,encoding="utf8",errors="replace").read().split("\n")
    h=next(i for i,l in enumerate(L) if "OpType" in l and "Target" in l)
    rows=[]
    for l in L[h+1:]:
        m=re.search(r"\] (\d+)\s+(\w+)\s+(\w+)\s+(NPU|CPU)\s+\S+\s+\S+\s+(\d+)/(\d+)/(\d+)\s+(\d+)\s+(\S+)",l)
        if m: rows.append(dict(id=int(m.group(1)),op=m.group(2),tgt=m.group(4),ddr=int(m.group(5)),npu=int(m.group(6)),tot=int(m.group(7)),time=int(m.group(8)),name=m.group(9)))
        elif rows: break
    return rows
for m in ("baseline","relu","resize_hp"):
    r=table(f"{S}/c5_{m}/stdout.txt")
    tot=sum(x["tot"] for x in r)
    by=collections.defaultdict(lambda:[0,0,0])
    for x in r:
        b=by[x["op"]]; b[0]+=1; b[1]+=x["tot"]; b[2]+=x["time"]
    print(f"== {m}: rows {len(r)} total cycles {tot}")
    for op,(n,c,t) in sorted(by.items(),key=lambda kv:-kv[1][1]):
        print(f"  {op:14s} n={n:3d} cycles={c:9d} share={100*c/tot:5.1f}%  time_col_sum={t}")
    if m!="relu":
        print("  resize rows:",[(x["id"],x["tgt"],x["ddr"],x["npu"],x["tot"],x["time"]) for x in r if x["op"]=="Resize"])
