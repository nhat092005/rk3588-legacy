import onnx,sys
m=onnx.load(sys.argv[1])
for n in m.graph.node:
    if n.op_type=="Resize":
        for a in n.attribute:
            if a.name=="coordinate_transformation_mode": a.s=b"half_pixel"
            if a.name=="nearest_mode": a.s=b"round_prefer_floor"
onnx.save(m,sys.argv[2])
