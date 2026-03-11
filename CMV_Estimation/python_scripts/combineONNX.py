
# Python 3 code
import os
import onnx
from onnx import external_data_helper

fn = './results/raft_kubric_pretrained.onnx'
base_dir = os.path.dirname(fn)  # folder containing .onnx and external data files

model = onnx.load(fn, load_external_data=False)   # load without auto external load
external_data_helper.load_external_data_for_model(model, base_dir)
onnx.save_model(model, 'results/raft_kubric_pretrained_merged.onnx')

