"""Shared helpers for the OpenVINO 2020.3/Movidius examples."""
from .request_pool import InferenceResult, RequestPool
from .device import DeviceSpec, parse_device_spec, resolve_servers
