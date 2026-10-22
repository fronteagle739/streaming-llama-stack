#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../engines/ds4-mac/ds4_metal_wired.h"

int main(int argc,char **argv) {
    if(argc!=2 || (strcmp(argv[1],"admitted") && strcmp(argv[1],"refused")))return 2;
    const char *budget=getenv("DS4_NATIVE_SESSION_HOST_BYTES");
    const char *floor=getenv("DS4_NATIVE_SESSION_HOST_FLOOR_BYTES");
    const char *wired=getenv("DS4_METAL_WIRED_CORE_KV");
    if(!budget || !floor || !wired)return 3;
    printf("{\"host_bytes\":\"%s\",\"floor_bytes\":\"%s\",\"wired\":\"%s\"}\n",budget,floor,wired);
    fflush(stdout);
    @autoreleasepool {
        id<MTLDevice> device=MTLCreateSystemDefaultDevice();
        if(!device)return 4;
        @autoreleasepool {
            id<MTLBuffer> buffer=ds4_metal_owned_wired_buffer(device,1048576,MTLResourceStorageModeShared,NULL);
            const bool expected=!strcmp(argv[1],"admitted");
            if((buffer!=nil)!=expected)return 5;
            if(buffer){
                ((unsigned char *)buffer.contents)[0]=42;
                if(((unsigned char *)buffer.contents)[0]!=42 || atomic_load(&ds4_wired_live_bytes)<1048576)return 6;
            }
            printf("MAC_FACTORY_%s\n",buffer?"ADMITTED":"REFUSED");
        }
        if(atomic_load(&ds4_wired_live_bytes))return 7;
    }
    return 0;
}
