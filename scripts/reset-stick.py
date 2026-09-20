#!/usr/bin/env python3
"""Put the MA2450 stick back into ROM/boot mode (03e7:2150) without unplugging it.

    sudo python3 scripts/reset-stick.py            # USBDEVFS_RESET, then poll
    sudo python3 scripts/reset-stick.py --power    # VBUS cycle of the port first

Run this on the **host** (needs write access to /dev/bus/usb and to /sys).  A run
that failed after firmware boot leaves the stick as 03e7:f63b, which then looks
like "stick not detected" on the next attempt - this is the recovery.

Adapted from the tool used in the sibling NCSDK project
(/home/chris/workspace/flash-movidius/tools/ma2reset.py).
"""
import errno
import fcntl
import glob
import os
import sys
import time

USBDEVFS_RESET = 21780
VENDOR = "03e7"
BOOT, RUN = "2150", "f63b"


def scan():
    """Return (devpath, portdir, pid) for the first 03e7 device, or None."""
    for f in glob.glob("/sys/bus/usb/devices/*/idVendor"):
        try:
            if open(f).read().strip() != VENDOR:
                continue
        except OSError:
            continue
        d = os.path.dirname(f)
        pid = open(os.path.join(d, "idProduct")).read().strip()
        bus = int(open(os.path.join(d, "busnum")).read().strip())
        dev = int(open(os.path.join(d, "devnum")).read().strip())
        return "/dev/bus/usb/%03d/%03d" % (bus, dev), d, pid
    return None


def wait(pid_want, seconds=25):
    for _ in range(seconds * 2):
        s = scan()
        if s and s[2] == pid_want:
            return s
        time.sleep(0.5)
    return None


power = "--power" in sys.argv
seen = scan()
if not seen:
    raise SystemExit("no 03e7 device present")
path, portdir, pid = seen
print("found 03e7:%s at %s (port %s)" % (pid, path, os.path.basename(portdir)))

if power or pid == RUN:
    auth = os.path.join(portdir, "authorized")
    if os.path.exists(auth):
        open(auth, "w").write("0")
        print("  port deauthorized (VBUS off)")
        time.sleep(2)
        open(auth, "w").write("1")
        print("  port reauthorized (VBUS on)")
        time.sleep(1)

if os.path.exists(path):
    # Resetting a *booted* stick tears the device down underneath this very call,
    # so the ioctl usually returns ENODEV/ENXIO even though the reset worked and
    # the stick comes back in boot mode.  Treat that as normal and let the poll
    # decide; bailing out here would escalate needlessly to a VBUS cycle.
    fd = os.open(path, os.O_WRONLY)
    try:
        fcntl.ioctl(fd, USBDEVFS_RESET, 0)
        print("  USBDEVFS_RESET issued")
    except OSError as ex:
        if ex.errno in (errno.ENODEV, errno.ENXIO, errno.ENOENT, errno.EIO, errno.ETIMEDOUT):
            print("  USBDEVFS_RESET -> %s (device vanished mid-reset: normal for a booted stick)"
                  % errno.errorcode.get(ex.errno, ex.errno))
        else:
            raise
    finally:
        try:
            os.close(fd)
        except OSError:
            pass

got = wait(BOOT)
if got:
    print("RESULT: in boot mode 03e7:%s at %s" % (BOOT, got[0]))
    sys.exit(0)

got = scan()
print("RESULT: not in boot mode; currently %s" % (("03e7:" + got[2]) if got else "absent"))
print("        retry with: sudo python3 scripts/reset-stick.py --power")
sys.exit(1)
