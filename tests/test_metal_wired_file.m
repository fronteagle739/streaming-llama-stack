/* Actual Metal file-to-wired copy, including a chunk boundary and EOF padding. */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <assert.h>
#include <fcntl.h>
#include <mach/mach_vm.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../engines/ds4-mac/ds4_metal_wired.h"

static unsigned char pattern(size_t i) { return (unsigned char)(i*13+7); }
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

int main(int argc,char **argv) {
    assert(argc==2);
    setenv("DS4_METAL_WIRED_CORE_KV","1",1);
    setenv("DS4_NATIVE_SESSION_HOST_BYTES","536870912",1);
    setenv("DS4_NATIVE_SESSION_HOST_FLOOR_BYTES","34359738368",1);
    const size_t bytes=96u*1024u*1024u,file_bytes=bytes+17;
    int fd=open(argv[1],O_RDWR|O_CREAT|O_EXCL,0600);assert(fd>=0);
    unsigned char block[65536];
    for(size_t pos=0;pos<file_bytes;) {
        size_t n=file_bytes-pos;if(n>sizeof(block))n=sizeof(block);
        for(size_t i=0;i<n;i++)block[i]=pattern(pos+i);
        assert(write(fd,block,n)==(ssize_t)n);pos+=n;
    }
    assert(fsync(fd)==0);
    @autoreleasepool {
        id<MTLDevice> device=MTLCreateSystemDefaultDevice();assert(device);
        @autoreleasepool {
            id<MTLBuffer> buffer=ds4_metal_owned_wired_file_buffer(device,bytes,MTLResourceStorageModeShared,fd,17);
            assert(buffer);check_region(buffer.contents);
            const unsigned char *data=buffer.contents;
            for(size_t i=0;i<bytes;i++)assert(data[i]==pattern(i+17));
            assert(atomic_load(&ds4_wired_live_bytes)==bytes);
        }
        assert(atomic_load(&ds4_wired_live_bytes)==0);
        @autoreleasepool {
            size_t page=(size_t)getpagesize();
            id<MTLBuffer> tail=ds4_metal_owned_wired_file_buffer(device,page,MTLResourceStorageModeShared,fd,bytes);
            assert(tail);check_region(tail.contents);
            const unsigned char *data=tail.contents;
            for(size_t i=0;i<page;i++)assert(data[i]==(i<17?pattern(bytes+i):0));
            assert(!ds4_metal_owned_wired_file_buffer(device,page*2,MTLResourceStorageModeShared,fd,bytes));
        }
        assert(atomic_load(&ds4_wired_live_bytes)==0);
        assert(!ds4_metal_owned_wired_file_buffer(device,16384,MTLResourceStorageModeShared,-1,0));
        setenv("DS4_NATIVE_SESSION_HOST_BYTES","1",1);
        assert(!ds4_metal_owned_wired_file_buffer(device,16384,MTLResourceStorageModeShared,fd,0));
        assert(atomic_load(&ds4_wired_live_bytes)==0);
    }
    close(fd);
    puts("PASS: 96MiB exact offset copy across chunk boundary; anonymous wired storage; EOF zero padding; range/descriptor/budget refusal; release");
    return 0;
}
