# ONNX Fixtures

`external-chain.onnx` is a repo-local benchmark smoke fixture for indexed-session
validation. It contains a 1,024-node linear ONNX graph and one float32
initializer stored as external data in `external-chain.bin`.

The external payload is intentionally larger than the protobuf so CI can verify
that ONNX external-data descriptors are covered without requiring payload bytes
to be loaded on open.
