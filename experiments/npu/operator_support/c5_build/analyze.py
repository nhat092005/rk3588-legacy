import re,collections,onnx
S="/tmp/claude-1000/-home-nhat-workspace-prjs-rk3588/d98a5996-660b-48b1-8f49-5240c06786b7/scratchpad"
def table(path):
    L=open(path,encoding="utf8",errors="replace").read().split("\n")
    h=next(i for i,l in enumerate(L) if "OpType" in l and "Target" in l)
    rows=[]
    for l in L[h+1:]:
        m=re.search(r"\] (\d+)\s+(\w+)\s+(\w+)\s+(NPU|CPU)\s+\S+\s+\S+\s+\S+\s+(\d+)\s+(\S+)",l)
        if m: rows.append((int(m.group(1)),m.group(2),m.group(4)))
        elif rows: break
    return rows
for m in ("baseline","relu"):
    r=table(f"{S}/c5_{m}/stdout.txt")
    print(m,"rows",len(r),dict(collections.Counter(x[1] for x in r).most_common()))
    print("  CPU:",[(x[0],x[1]) for x in r if x[2]=="CPU"])
    o=onnx.load(f"{S}/c5_{m}/best.onnx")
    print("  onnx",len(o.graph.node),dict(collections.Counter(n.op_type for n in o.graph.node).most_common()),[[d.dim_value for d in x.type.tensor_type.shape.dim] for x in o.graph.output])
    rs=[n for n in o.graph.node if n.op_type=="Resize"]
    print("  resize attrs:",[(a.name,a.s.decode() if a.s else a.i) for a in rs[0].attribute if a.name in("mode","coordinate_transformation_mode","nearest_mode")])
    log=open(f"{S}/c5_{m}/toolkit.log",encoding="utf8",errors="replace").read()
    print("  replace_exswish:",log.count("replace_exswish:"),"| passes:",sorted(set(re.findall(r"D     (\w+):",log))))
