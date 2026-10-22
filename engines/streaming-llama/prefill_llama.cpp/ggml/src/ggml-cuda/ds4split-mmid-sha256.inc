// Copied unchanged from 196889b86f0d01a4074cd2f9054af47e8abb193e src/boundary-capture.cpp
uint32_t rotr(uint32_t v, unsigned n) { return (v >> n) | (v << (32 - n)); }
struct digest {
    uint32_t h[8] = {0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,
                     0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
    std::array<unsigned char,64> block{};
    uint64_t total = 0;
    size_t pending = 0;
    void transform() {
        static const uint32_t k[64] = {
            0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
            0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
            0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
            0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
            0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
            0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
            0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
            0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};
        uint32_t w[64];
        for (unsigned i=0;i<16;++i) w[i]=(uint32_t(block[4*i])<<24)|(uint32_t(block[4*i+1])<<16)|
            (uint32_t(block[4*i+2])<<8)|block[4*i+3];
        for (unsigned i=16;i<64;++i) {
            uint32_t a=w[i-15],b=w[i-2];
            w[i]=w[i-16]+(rotr(a,7)^rotr(a,18)^(a>>3))+w[i-7]+(rotr(b,17)^rotr(b,19)^(b>>10));
        }
        uint32_t a=h[0],b=h[1],c=h[2],d=h[3],e=h[4],f=h[5],g=h[6],z=h[7];
        for (unsigned i=0;i<64;++i) {
            uint32_t x=z+(rotr(e,6)^rotr(e,11)^rotr(e,25))+((e&f)^(~e&g))+k[i]+w[i];
            uint32_t y=(rotr(a,2)^rotr(a,13)^rotr(a,22))+((a&b)^(a&c)^(b&c));
            z=g;g=f;f=e;e=d+x;d=c;c=b;b=a;a=x+y;
        }
        h[0]+=a;h[1]+=b;h[2]+=c;h[3]+=d;h[4]+=e;h[5]+=f;h[6]+=g;h[7]+=z;
    }
    void update(const void *input, size_t n) {
        auto *data=static_cast<const unsigned char *>(input);
        require(n<=UINT64_MAX-total,"digest size overflow");total+=n;
        while(n) {size_t take=std::min(n,64-pending);std::memcpy(block.data()+pending,data,take);
            pending+=take;data+=take;n-=take;if(pending==64){transform();pending=0;}}
    }
    std::string finish() {
        require(total<=UINT64_MAX/8,"digest bit count overflow");uint64_t bits=total*8;
        unsigned char one=0x80,zero=0;update(&one,1);
        while(pending!=56)update(&zero,1);
        unsigned char end[8];for(unsigned i=0;i<8;++i)end[7-i]=(unsigned char)(bits>>(8*i));update(end,8);
        char out[65];for(unsigned i=0;i<8;++i)std::snprintf(out+8*i,9,"%08x",h[i]);return out;
    }
};
