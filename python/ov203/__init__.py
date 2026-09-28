"""Shared helpers for the OpenVINO 2020.3/Movidius examples."""
from .camera import FakeCamera, LatestFrame
from .device import DeviceSpec, parse_device_spec, resolve_servers
from .display import DisplayPump, WindowWatcher
from .process import ProcCpu, spawn_servers, stop_server, wait_alive
from .request_pool import InferenceResult, RequestPool
from .runtime import models_root, source_repo_root, standalone_root

__all__ = [
    "DeviceSpec", "DisplayPump", "FakeCamera", "InferenceResult", "LatestFrame",
    "ProcCpu", "RequestPool", "WindowWatcher", "models_root",
    "parse_device_spec", "resolve_servers", "source_repo_root", "spawn_servers",
    "standalone_root", "stop_server", "wait_alive",
]
