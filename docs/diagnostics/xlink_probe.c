/* Throwaway diagnostic (not part of the deliverable).
 *
 * Calls the vendored OpenVINO 2020.3.2 XLink search/boot API directly and prints
 * every device name + count it sees before and after XLinkBoot(), which is exactly
 * what mvnc's ncDeviceOpen() does internally when it looks for "the booted device".
 */
#include <XLink.h>
#include <XLinkLog.h>

#include <stdio.h>
#include <string.h>
#include <unistd.h>

static void dump(const char *tag, XLinkError_t rc, const deviceDesc_t *arr, unsigned int n) {
    printf("%-28s rc=%d n=%u\n", tag, (int)rc, n);
    for (unsigned int i = 0; i < n; ++i) {
        printf("      [%u] name='%s' protocol=%d platform=%d\n", i, arr[i].name,
               (int)arr[i].protocol, (int)arr[i].platform);
    }
}

int main(void) {
    XLinkGlobalHandler_t global;
    memset(&global, 0, sizeof(global));
    printf("XLinkInitialize -> %d\n", (int)XLinkInitialize(&global));

    deviceDesc_t any;
    memset(&any, 0, sizeof(any));
    any.protocol = X_LINK_USB_VSC;
    any.platform = X_LINK_ANY_PLATFORM;

    deviceDesc_t arr[8];
    unsigned int n = 0;

    memset(arr, 0, sizeof(arr));
    n = 0;
    XLinkError_t rc = XLinkFindAllSuitableDevices(X_LINK_ANY_STATE, any, arr, 8, &n);
    dump("BEFORE any_state(ANY)", rc, arr, n);

    memset(arr, 0, sizeof(arr));
    n = 0;
    rc = XLinkFindAllSuitableDevices(X_LINK_UNBOOTED, any, arr, 8, &n);
    dump("BEFORE unbooted(ANY)", rc, arr, n);

    deviceDesc_t boot;
    memset(&boot, 0, sizeof(boot));
    rc = XLinkFindFirstSuitableDevice(X_LINK_UNBOOTED, any, &boot);
    printf("FIRST unbooted -> rc=%d name='%s' platform=%d protocol=%d\n", (int)rc, boot.name,
           (int)boot.platform, (int)boot.protocol);
    if (rc != X_LINK_SUCCESS) {
        printf("no unbooted device, aborting\n");
        return 2;
    }

    printf("XLinkBoot(%s, /out/usb-ma2450.mvcmd) ...\n", boot.name);
    rc = XLinkBoot(&boot, "/out/usb-ma2450.mvcmd");
    printf("XLinkBoot -> %d\n", (int)rc);

    char bootname[128];
    snprintf(bootname, sizeof(bootname), "%s", boot.name);

    for (int k = 0; k < 12; ++k) {
        sleep(1);
        memset(arr, 0, sizeof(arr));
        n = 0;
        rc = XLinkFindAllSuitableDevices(X_LINK_ANY_STATE, any, arr, 8, &n);
        dump("AFTER any_state(ANY)", rc, arr, n);

        memset(arr, 0, sizeof(arr));
        n = 0;
        rc = XLinkFindAllSuitableDevices(X_LINK_BOOTED, any, arr, 8, &n);
        dump("AFTER booted(ANY)", rc, arr, n);

        deviceDesc_t byname;
        memset(&byname, 0, sizeof(byname));
        byname.protocol = X_LINK_USB_VSC;
        byname.platform = X_LINK_ANY_PLATFORM;
        snprintf(byname.name, sizeof(byname.name), "%s", bootname);
        deviceDesc_t found;
        memset(&found, 0, sizeof(found));
        rc = XLinkFindFirstSuitableDevice(X_LINK_ANY_STATE, byname, &found);
        printf("AFTER find-by-old-name '%s' -> rc=%d found='%s'\n", byname.name, (int)rc,
               found.name);
    }


    return 0;
}
