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
#include <sys/stat.h>
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
static id<MTLBuffer> ds4_metal_owned_wired_buffer_impl(id<MTLDevice> device,
        NSUInteger bytes,MTLResourceOptions options,const void *source,
        int source_fd,uint64_t source_offset) {
    uint64_t page=(uint64_t)getpagesize();
    if(!bytes||bytes>SIZE_MAX-(page-1)||!ds4_wired_admit(bytes)) {
        fprintf(stderr,"ds4: owned Metal allocation refused by physical host budget\n");return nil;
    }
    size_t length=(size_t)((bytes+page-1)&~(page-1));
    void *memory=mmap(NULL,length,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANON,-1,0);
    if(memory==MAP_FAILED)return nil;
    if(source_fd>=0 && (source_offset>INT64_MAX || bytes>INT64_MAX-source_offset)) {
        munmap(memory,length);return nil;
    }
    uint64_t source_bytes=bytes;
    if(source_fd>=0) {
        struct stat info;
        if(fstat(source_fd,&info)!=0 || !S_ISREG(info.st_mode) || info.st_size<0 ||
           (uint64_t)info.st_size>UINT64_MAX-(page-1) || source_offset>(uint64_t)info.st_size ||
           bytes>(((uint64_t)info.st_size+page-1)&~(page-1))-source_offset) {
            fprintf(stderr,"ds4: owned Metal model range exceeds its input file\n");
            munmap(memory,length);return nil;
        }
        if(source_bytes>(uint64_t)info.st_size-source_offset)
            source_bytes=(uint64_t)info.st_size-source_offset;
    }
    /* Bound both VM operations and input reads. A whole-model memcpy from an
     * mmap keeps a second resident view of the weights during startup. Reading
     * the existing descriptor directly fills only our owned, wired storage. */
    const size_t chunk_limit=64u*1024u*1024u;
    for(size_t pos=0;pos<length;) {
        size_t chunk=length-pos;if(chunk>chunk_limit)chunk=chunk_limit;
        if(!ds4_wired_admit(chunk) || mlock((char *)memory+pos,chunk)!=0) {
            fprintf(stderr,"ds4: owned Metal chunk refused or mlock failed at %zu: %s\n",pos,strerror(errno));
            munlock(memory,length);munmap(memory,length);return nil;
        }
        size_t copy=pos<source_bytes?(size_t)(source_bytes-pos):0;if(copy>chunk)copy=chunk;
        if(source_fd>=0) {
            size_t done=0;
            while(done<copy) {
                ssize_t got=pread(source_fd,(char *)memory+pos+done,copy-done,
                                  (off_t)(source_offset+pos+done));
                if(got<0 && errno==EINTR)continue;
                if(got<=0) {
                    fprintf(stderr,"ds4: owned Metal model read failed at %zu: %s\n",pos+done,
                            got==0?"unexpected end of file":strerror(errno));
                    munlock(memory,length);munmap(memory,length);return nil;
                }
                done+=(size_t)got;
            }
        } else if(source)memcpy((char *)memory+pos,(const char *)source+pos,copy);
        pos+=chunk;
        if(source_fd>=0 && (pos==length || pos%(4ull*1024*1024*1024)==0))
            fprintf(stderr,"ds4: owned wired model read %zu/%zu bytes\n",pos,length);
    }
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
            length,atomic_load(&ds4_wired_live_bytes),source_fd>=0?"file-read":source?"core-copy":"allocation");
    return result;
}
static id<MTLBuffer> ds4_metal_owned_wired_buffer(id<MTLDevice> device,
        NSUInteger bytes,MTLResourceOptions options,const void *source) {
    return ds4_metal_owned_wired_buffer_impl(device,bytes,options,source,-1,0);
}
static id<MTLBuffer> ds4_metal_owned_wired_file_buffer(id<MTLDevice> device,
        NSUInteger bytes,MTLResourceOptions options,int fd,uint64_t offset) {
    if(fd<0) {
        fprintf(stderr,"ds4: owned Metal model descriptor is not registered\n");return nil;
    }
    return ds4_metal_owned_wired_buffer_impl(device,bytes,options,NULL,fd,offset);
}
#endif
