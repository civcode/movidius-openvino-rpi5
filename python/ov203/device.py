from dataclasses import dataclass

@dataclass(frozen=True)
class DeviceSpec:
    requested: str
    uses_myriad: bool
    uses_cpu: bool
    physical_devices: tuple

def parse_device_spec(requested):
    if not requested:
        raise ValueError("device must not be empty")
    tokens = []
    cur = []
    for ch in requested.upper():
        if ch in ":," or ch.isspace():
            if cur:
                tokens.append("".join(cur))
                cur = []
        else:
            cur.append(ch)
    if cur:
        tokens.append("".join(cur))
    physical = []
    for token in tokens:
        if token not in ("HETERO", "MULTI") and token not in physical:
            physical.append(token)
    if not physical:
        raise ValueError("device specification has no physical plugin")
    return DeviceSpec(requested, "MYRIAD" in physical, "CPU" in physical, tuple(sorted(physical)))

def resolve_servers(requested, device):
    if requested < 1:
        raise ValueError("servers must be >= 1")
    return requested if parse_device_spec(device).uses_cpu else 1
