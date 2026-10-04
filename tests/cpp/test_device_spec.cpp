#include <ov203/device_spec.hpp>

#include <cassert>
#include <stdexcept>

int main() {
    {
        const auto spec = ov203::parseDeviceSpec("MYRIAD");
        assert(spec.usesMyriad);
        assert(!spec.usesCpu);
        assert(spec.physicalDevices.size() == 1);
        assert(spec.physicalDevices[0] == "MYRIAD");
    }
    {
        const auto spec = ov203::parseDeviceSpec("HETERO:CPU,MYRIAD");
        assert(spec.usesMyriad);
        assert(spec.usesCpu);
        assert(spec.physicalDevices.size() == 2);
        assert(spec.physicalDevices[0] == "CPU");
        assert(spec.physicalDevices[1] == "MYRIAD");
    }
    {
        bool threw = false;
        try {
            (void)ov203::parseDeviceSpec("HETERO");
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        assert(threw);
    }
    return 0;
}
