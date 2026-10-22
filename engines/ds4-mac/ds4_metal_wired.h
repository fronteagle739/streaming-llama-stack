/* Owned shared-Metal storage for core and KV. Allocation/locking follows the
 * native MiMo wired-buffer pattern at c427e423702145aeb8b9e14d05f1c60cee31d9b9.
 * No tensor arithmetic or native state format changes. */
#ifndef DS4_METAL_WIRED_H
#define DS4_METAL_WIRED_H
#include <errno.h>
#include <inttypes.h>
#include <mach/mach.h>
#include <stdatomic.h>
#include <sys/mman.h>
#include <unistd.h>

static atomic_uint_fast64_t ds4_wired_live_bytes;
static int ds4_metal_wired_enabled(void) {
    const char *s=getenv("DS4_METAL_WIRED_CORE_KV");
    return s && strcmp(s,"1")==0;
}
static int ds4_wired_limit(const char *name,uint64_t *value) {
    const char *s=getenv(name);if(!s||!*s)return 0;
    uint64_t v=0;
    for(;*s;s++) { if(*s<'0'||*s>'9'||v>(UINT64_MAX-(*s-'0'))/10)return 0;v=v*10+(*s-'0'); }
    *value=v;return v!=0;
}
static int ds4_wired_admit(uint64_t bytes) {
    uint64_t cap=0,floor=0;
    if(!ds4_wired_limit("DS4_NATIVE_SESSION_HOST_BYTES",&cap)||
       !ds4_wired_limit("DS4_NATIVE_SESSION_HOST_FLOOR_BYTES",&floor))return 0;
    vm_statistics64_data_t vm;mach_msg_type_number_t n=HOST_VM_INFO64_COUNT;
    mach_port_t host=mach_host_self();vm_size_t page=0;
    kern_return_t a=host_page_size(host,&page);
    kern_return_t b=host_statistics64(host,HOST_VM_INFO64,(host_info64_t)&vm,&n);
    mach_port_deallocate(mach_task_self(),host);
    mach_task_basic_info_data_t task;n=MACH_TASK_BASIC_INFO_COUNT;
    kern_return_t c=task_info(mach_task_self(),MACH_TASK_BASIC_INFO,(task_info_t)&task,&n);
    if(a!=KERN_SUCCESS||b!=KERN_SUCCESS||c!=KERN_SUCCESS)return 0;
    uint64_t reclaim=(uint64_t)vm.inactive_count+vm.speculative_count;
    if((uint64_t)vm.external_page_count>reclaim)reclaim=vm.external_page_count;
    uint64_t available=((uint64_t)vm.free_count+reclaim)*page;
    return task.resident_size<=cap && bytes<=cap-task.resident_size &&
           available>=floor && bytes<=available-floor;
}
static id<MTLBuffer> ds4_metal_owned_wired_buffer(id<MTLDevice> device,
        NSUInteger bytes,MTLResourceOptions options,const void *source) {
    uint64_t page=(uint64_t)getpagesize();
    if(!bytes||bytes>SIZE_MAX-(page-1)||!ds4_wired_admit(bytes)) {
        fprintf(stderr,"ds4: owned Metal allocation refused by physical host budget\n");return nil;
    }
    size_t length=(size_t)((bytes+page-1)&~(page-1));
    void *memory=mmap(NULL,length,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANON,-1,0);
    if(memory==MAP_FAILED)return nil;
    if(mlock(memory,length)!=0) {
        fprintf(stderr,"ds4: owned Metal mlock failed (%zu bytes): %s\n",length,strerror(errno));
        munmap(memory,length);return nil;
    }
    if(source)memcpy(memory,source,bytes);
    atomic_fetch_add(&ds4_wired_live_bytes,length);
    id<MTLBuffer> result=[device newBufferWithBytesNoCopy:memory length:length
        options:((options & ~MTLResourceStorageModeMask)|MTLResourceStorageModeShared)
        deallocator:^(void *pointer,NSUInteger count) {
            munlock(pointer,count);munmap(pointer,count);
            atomic_fetch_sub(&ds4_wired_live_bytes,count);
        }];
    if(!result) {
        atomic_fetch_sub(&ds4_wired_live_bytes,length);munlock(memory,length);munmap(memory,length);
        return nil;
    }
    fprintf(stderr,"ds4: owned anonymous wired Metal buffer=%zu live=%"PRIuFAST64" source=%s\n",
            length,atomic_load(&ds4_wired_live_bytes),source?"core-copy":"allocation");
    return result;
}
#endif
