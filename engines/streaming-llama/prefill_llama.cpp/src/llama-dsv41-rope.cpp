#include "llama-dsv41-rope.h"

#include <cstdint>
#include <cstring>

std::array<float, 32> llama_dsv41_rope_frequencies(bool compressed) {
    // stack-backend c8aa547e03ff14839b32763045a2c86f2e8362b5,
    // ds4_metal.m49969-49984. Generated on the target Mac with the production
    // Objective-C dispatch_once block and -O3 -ffast-math -mcpu=native flags.
    // Strict flags independently produced identical 64 words. Production blob
    // SHA256: 4673dafa7727ad2ebd5475c2b1f094acede2bf4eab26bb51de30f1da59267a1e.
    // Row 0: base 10000, no YaRN. Row 1: base 160000, YaRN factor 16.
    static constexpr uint32_t bits[2][32] = {
        {1065353216,1061157137,1058010522,1054337179,1050798235,1047712803,1043732615,1040747895,
         1036831949,1033475085,1030116802,1026341397,1023510243,1019364175,1016180025,1012562930,
         1008981770,1005959600,1001931932,998911607,995049016,991652109,988353832,984533390,
         981668463,977573004,974350870,970790693,967166816,964208661,960132947,957076594},
        {1065353216,1060112954,1056054303,1051098369,1046804782,1042117749,1037601972,1033169251,
         1028443341,1024251131,1019326489,1015361742,1010249151,1006366088,1001209182,997081655,
         991197921,985076960,979287478,973418702,966505391,959469944,952012877,943627103,
         933484300,917783504,913828774,908753896,904558591,899759068,895336240,890797133}
    };
    static_assert(sizeof(float) == sizeof(uint32_t), "native frequency word size");
    std::array<float, 32> result;
    std::memcpy(result.data(), bits[compressed ? 1 : 0], sizeof(result));
    return result;
}
