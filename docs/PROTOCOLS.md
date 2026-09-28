# Inference server wire protocols

All stream protocols are binary-safe stdio protocols. Multi-byte integers and
floating-point values are **little-endian**. EOF between complete requests ends
a stream normally; EOF inside a frame or response is an error. Current protocol
version is `1` and is defined by this document; no handshake is sent yet.

## MobileNet classification

Input is one fixed-size tensor per request: `1x3x224x224` float32 values in NCHW
order, little-endian. Output is exactly 1000 float32 logits, little-endian. Batch
size is one. The server validates a rank-4, three-channel input and 1000 output
elements before serving requests.

## SSDLite detection

Each request is:

```text
uint32_le width
uint32_le height
uint8 RGB[width * height * 3]    # interleaved HWC
```

`width` and `height` must be in `1..8192`. The response is ASCII, terminated by
`END`:

```text
FRAME <width> <height> <infer_ms>
DET <label> <score> <x1> <y1> <x2> <y2>   # zero or more
END
```

Labels containing spaces are encoded/parsed according to the existing detector
client rules.

## DeepLab segmentation

There are three distinct channel/layout layers:

```text
Camera/OpenCV representation: BGR HWC uint8
Wire representation:          RGB HWC uint8
IR input blob:                BGR NCHW raw 0..255
```

The generated DeepLab IR contains the preprocessing/channel reversal needed by
the original TensorFlow model. Each wire request is the same `width`, `height`,
then interleaved RGB bytes used by SSD.

The response is:

```text
FRAME <width> <height> <total_ms> <infer_ms>
CLASSES <n>
CLASS <id> <name> <pixels>                # n lines
MASK <width> <height>
<uint16_le class_id[width * height]>
END
```

Class-map files written with `--mask-out` are **P5 PGM**, one unsigned byte per
pixel, not P6 PPM. Class IDs above 255 cannot be represented by that export
format and are rejected.

## Compatibility rules

Wire encoding must never depend on the host CPU's native byte order. C++ code
uses `include/ov203/wire.hpp`; Python uses explicit little-endian `struct` or
NumPy dtypes. If framing changes incompatibly, add a versioned handshake rather
than changing this version silently.
