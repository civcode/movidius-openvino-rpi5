#include "include/ov203/wire.hpp"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <sstream>
#include <string>

int main() {
    {
        std::ostringstream out(std::ios::binary);
        ov203::writeLe16(out, 0x1234u);
        ov203::writeLe32(out, 0x89abcdefu);
        const std::string bytes = out.str();
        assert(bytes.size() == 6);
        assert(static_cast<unsigned char>(bytes[0]) == 0x34);
        assert(static_cast<unsigned char>(bytes[1]) == 0x12);
        assert(static_cast<unsigned char>(bytes[2]) == 0xef);
        assert(static_cast<unsigned char>(bytes[3]) == 0xcd);
        assert(static_cast<unsigned char>(bytes[4]) == 0xab);
        assert(static_cast<unsigned char>(bytes[5]) == 0x89);
    }

    {
        const float values[] = {1.0f, -2.5f, 0.0f};
        unsigned char bytes[sizeof(values)] = {};
        ov203::nativeFloat32ToLittleEndianBytes(values, bytes, 3);
        const unsigned char one[] = {0x00, 0x00, 0x80, 0x3f};
        assert(std::memcmp(bytes, one, 4) == 0);

        float decoded[3] = {};
        std::memcpy(decoded, bytes, sizeof(bytes));
        ov203::littleEndianFloat32ToNativeInPlace(decoded, 3);
        for (int i = 0; i < 3; ++i)
            assert(std::fabs(decoded[i] - values[i]) < 1e-6f);
    }

    {
        const char raw[] = {char(0x78), char(0x56), char(0x34), char(0x12)};
        std::istringstream in(std::string(raw, sizeof(raw)), std::ios::binary);
        assert(ov203::readLe32(in) == 0x12345678u);
    }

    return 0;
}
