#!/bin/bash

set -ev

export PYTHONPATH=$HOME/onnxruntime/build_rocm/Release/build/lib

python onnxruntime/python/tools/transformers/models/llama2/llama-v2.py \
  --model meta-llama/Llama-2-7b-hf \
  --ort \
  --generate \
  --tunable \
  --tuning \
  -dm  llama2-7b-hf/rank-0_decoder_model_fp8.onnx \
  -dpm llama2-7b-hf/rank-0_decoder_with_past_model_fp8.onnx
