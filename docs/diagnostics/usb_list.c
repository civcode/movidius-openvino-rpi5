/* Throwaway diagnostic: minimal libusb device lister.
   Prints bus/devnum/vid/pid/product-string for every USB device libusb sees. */
#include <libusb-1.0/libusb.h>
#include <stdio.h>

int main(void) {
    libusb_context *ctx = NULL;
    if (libusb_init(&ctx) < 0) { printf("libusb_init failed\n"); return 1; }
    const struct libusb_version *v = libusb_get_version();
    printf("libusb version: %d.%d.%d (%s)\n", v->major, v->minor, v->micro, v->describe);
    libusb_device **devs = NULL;
    int n = libusb_get_device_list(ctx, &devs);
    printf("libusb_get_device_list -> %d devices\n", n);
    for (int i = 0; i < n; ++i) {
        struct libusb_device_descriptor d;
        if (libusb_get_device_descriptor(devs[i], &d) < 0) continue;
        unsigned char ports[8];
        int pc = libusb_get_port_numbers(devs[i], ports, 8);
        printf("  bus=%03u dev=%03u %04x:%04xbcd=%04x ports=", libusb_get_bus_number(devs[i]),
               libusb_get_device_address(devs[i]), d.idVendor, d.idProduct, d.bcdDevice);
        for (int k = 0; k < pc; ++k) printf("%d%s", ports[k], k + 1 < pc ? "." : "");
        printf("\n");
    }
    libusb_free_device_list(devs, 1);
    libusb_exit(ctx);
    return 0;
}
