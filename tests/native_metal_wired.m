#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <assert.h>
#include <mach/mach_vm.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../engines/ds4-mac/ds4_metal_wired.h"
static void check_region(void *pointer) {
    mach_vm_address_t address=(mach_vm_address_t)pointer;mach_vm_size_t size=0;
    vm_region_extended_info_data_t info;mach_msg_type_number_t count=VM_REGION_EXTENDED_INFO_COUNT;
    mach_port_t object=MACH_PORT_NULL;
    assert(mach_vm_region(mach_task_self(),&address,&size,VM_REGION_EXTENDED_INFO,
        (vm_region_info_t)&info,&count,&object)==KERN_SUCCESS);
    if(object!=MACH_PORT_NULL)mach_port_deallocate(mach_task_self(),object);
    assert(!info.external_pager && info.pages_swapped_out==0);
    address=(mach_vm_address_t)pointer;
    vm_region_basic_info_data_64_t basic;count=VM_REGION_BASIC_INFO_COUNT_64;
    assert(mach_vm_region(mach_task_self(),&address,&size,VM_REGION_BASIC_INFO_64,
        (vm_region_info_t)&basic,&count,&object)==KERN_SUCCESS);
    if(object!=MACH_PORT_NULL)mach_port_deallocate(mach_task_self(),object);
    assert(basic.user_wired_count>0);
}
int main(void) {
    setenv("DS4_METAL_WIRED_CORE_KV","1",1);
    setenv("DS4_NATIVE_SESSION_HOST_BYTES","1073741824",1);
    setenv("DS4_NATIVE_SESSION_HOST_FLOOR_BYTES","4294967296",1);
    @autoreleasepool {
        id<MTLDevice> device=MTLCreateSystemDefaultDevice();assert(device);
        unsigned char source[16384];for(size_t i=0;i<sizeof(source);i++)source[i]=(i*13+7)&255;
        @autoreleasepool {
            id<MTLBuffer> buffer=ds4_metal_owned_wired_buffer(device,sizeof(source),MTLResourceStorageModeShared,source);
            assert(buffer && memcmp(buffer.contents,source,sizeof(source))==0);
            check_region(buffer.contents);assert(atomic_load(&ds4_wired_live_bytes)>=sizeof(source));
            ((unsigned char *)buffer.contents)[0]^=1;assert(source[0]==7);
        }
        assert(atomic_load(&ds4_wired_live_bytes)==0);
        setenv("DS4_NATIVE_SESSION_HOST_BYTES","1",1);
        assert(!ds4_metal_owned_wired_buffer(device,16384,MTLResourceStorageModeShared,NULL));
        unsetenv("DS4_NATIVE_SESSION_HOST_FLOOR_BYTES");
        assert(!ds4_metal_owned_wired_buffer(device,16384,MTLResourceStorageModeShared,NULL));
    }
    puts("PASS: exact core copy, anonymous region, wired count, no swap, separate storage, release and physical-budget refusal");
    return 0;
}
