# Standalone quick start

```bash
./setup.sh --check
./setup.sh                    # includes CPU fallback environments
./setup.sh --skip-cpu-fallback # lighter MYRIAD-only Python setup
./setup.sh --install-udev   # optional, requires sudo
./verify.sh
./bin/ov-device-list
```

The bundle is relocatable. All public commands resolve paths from their own location. Docker and the original source checkout are not required.

For included example models:

```bash
./bin/classify --headless --fake-camera --frames 10
./bin/detect --headless --file models/images/dog_ssd.ppm --frames 1
./bin/segment --headless --file models/images/dog_ssd.ppm --frames 1 --mask-out mask.pgm
```
