# Model development

The development profile contains OpenVINO 2020.3 Model Optimizer and the MYRIAD compiler.

Convert ONNX to IR:
```bash
./bin/mo-onnx --input_model model.onnx --data_type FP16 --output_dir build/ir
```

Convert TensorFlow to IR:
```bash
./bin/mo-tf --input_model frozen_graph.pb --data_type FP16 --output_dir build/ir [model-specific MO options]
```

Compile an IR for MYRIAD:
```bash
./bin/compile-model build/ir/model.xml --output build/model.blob
```

End to end:
```bash
./bin/build-model model.onnx --precision FP16 --output-dir build/model
```

Use direct `mo-*` invocation when a model needs framework-specific OpenVINO 2020.3 options such as TensorFlow Object Detection custom-operation configuration.
