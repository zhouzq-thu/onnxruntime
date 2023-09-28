#!/bin/bash

set -ev

export PYTHONPATH=$HOME/onnxruntime/build_rocm/Release/build/lib

python onnxruntime/python/tools/transformers/models/llama2/llama-v2.py \
  --model meta-llama/Llama-2-7b-hf \
  --output-name llama2-7b-hf \
  --ort \
  --export \
  --optimize \
  --generate \
  --save-opt \
  --tunable \
  --tuning \
  --convert-fp16
