# 2c-b INT8 diagnosis (simulator)

## silu
class cos 0.99087, box cos 0.99953, D3: lớp class mất nhiều hơn lớp box
counts: {'t0.25': {'golden_above': 47.9, 'missed': 1.0, 'extra': 1.5}, 't0.001': {'golden_above': 203.8, 'missed': 63.9, 'extra': 4.5}}
lowest single cos:
- Conv /model/model.22/cv3.1/cv3.1.2/Conv_output_0 0.97217
- Reshape /model/model.22/Reshape_4_output_0 0.97221
- Conv /model/model.22/cv3.2/cv3.2.2/Conv_output_0 0.97672
- Reshape /model/model.22/Reshape_5_output_0 0.97676
- Conv /model/model.22/cv3.0/cv3.0.2/Conv_output_0 0.98555
- Split /model/model.21/Split_output_0 0.99773
- Concat /model/model.20/Concat_output_0 0.99821
- Conv /model/model.1/conv/Conv_output_0 0.99830
- Split /model/model.8/Split_output_1 0.99834
- exSwish /model/model.21/cv1/act/Mul_output_0 0.99846

## relu
class cos 0.97819, box cos 0.99956, D3: lớp class mất nhiều hơn lớp box
counts: {'t0.25': {'golden_above': 49.6, 'missed': 2.7, 'extra': 1.8}, 't0.001': {'golden_above': 269.3, 'missed': 87.7, 'extra': 7.3}}
lowest single cos:
- Reshape /model/model.22/Reshape_4_output_0 0.96182
- Conv /model/model.22/cv3.1/cv3.1.2/Conv_output_0 0.96183
- Conv /model/model.22/cv3.2/cv3.2.2/Conv_output_0 0.97079
- Reshape /model/model.22/Reshape_5_output_0 0.97080
- Conv /model/model.22/cv3.0/cv3.0.2/Conv_output_0 0.98362
- Relu /model/model.1/act/Relu_output_0 0.99772
- Reshape /model/model.22/Sigmoid_output_0_rs 0.99867
- Sigmoid /model/model.22/Concat_1_output_0_sw 0.99870
- Relu /model/model.2/cv1/act/Relu_output_0 0.99877
- exSoftmax13 /model/model.22/dfl/Transpose_output_0_sw 0.99908

