/* Throwaway diagnostic (not part of the deliverable).
 *
 * Links OpenVINO 2020.3.2's own vendored mvnc + XLink sources and calls
 * ncAvailableDevices()/ncDeviceOpen() with NC_LOG_DEBUG, to see exactly which
 * USB device names the library sees before and after it boots the MA2450.
 */
#include <mvnc.h>
#include <watchdog.h>

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    int level = NC_LOG_DEBUG;
    ncGlobalSetOption(NC_RW_LOG_LEVEL, &level, sizeof(level));

    struct ncDeviceDescr_t list[8];
    memset(list, 0, sizeof(list));
    int count = 0;
    ncStatus_t st = ncAvailableDevices(list, 8, &count);
    printf("== ncAvailableDevices -> status=%d count=%d\n", (int)st, count);
    for (int i = 0; i < count; ++i) {
        printf("   [%d] protocol=%d platform=%d name='%s'\n", i, (int)list[i].protocol,
               (int)list[i].platform, list[i].name);
    }

    struct ncDeviceDescr_t desc = {0};
    if (count > 0) {
        desc = list[0];
    } else {
        desc.protocol = NC_ANY_PROTOCOL;
        desc.platform = NC_ANY_PLATFORM;
    }
    if (argc > 1) {
        snprintf(desc.name, sizeof(desc.name), "%s", argv[1]);
    }
    printf("== ncDeviceOpen name='%s' protocol=%d platform=%d\n", desc.name, (int)desc.protocol,
           (int)desc.platform);

    WatchdogHndl_t *wd = NULL;
    if (watchdog_create(&wd) != WD_ERRNO) {
        printf("watchdog_create failed\n");
        return 2;
    }

    ncDeviceOpenParams_t params;
    memset(&params, 0, sizeof(params));
    params.watchdogHndl = wd;
    params.watchdogInterval = 1000;  // OpenVINO 2020.3 myriad_config default
    params.customFirmwareDirectory = "";

    struct ncDeviceHandle_t *dev = NULL;
    st = ncDeviceOpen(&dev, desc, params);
    printf("== ncDeviceOpen -> %d\n", (int)st);
    if (st == NC_OK && dev) {
        ncDeviceClose(&dev, wd);
        printf("== ncDeviceClose OK\n");
    }
    watchdog_destroy(wd);
    return st != NC_OK;
}
