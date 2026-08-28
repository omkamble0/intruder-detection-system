# proto/v1 init
import sys
import os

# Ensure local import resolution for generated grpc stubs
proto_v1_dir = os.path.dirname(os.path.abspath(__file__))
if proto_v1_dir not in sys.path:
    sys.path.insert(0, proto_v1_dir)
