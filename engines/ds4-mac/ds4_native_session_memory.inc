#include <stdatomic.h>
#include <inttypes.h>
#include <mach/mach.h>
#include <sys/sysctl.h>
typedef struct {
    uint64_t budget, floor, physical;
    atomic_int stop, failed;
    pthread_t thread;
    int started;
} ds4_ns_memory_guard;

static int ds4_ns_memory_sample(uint64_t *available, uint64_t *resident) {
    vm_statistics64_data_t vm;mach_msg_type_number_t n=HOST_VM_INFO64_COUNT;
    mach_port_t host=mach_host_self();vm_size_t page=0;
    kern_return_t a=host_page_size(host,&page);
    kern_return_t b=host_statistics64(host,HOST_VM_INFO64,(host_info64_t)&vm,&n);
    mach_port_deallocate(mach_task_self(),host);
    mach_task_basic_info_data_t task; n=MACH_TASK_BASIC_INFO_COUNT;
    kern_return_t c=task_info(mach_task_self(),MACH_TASK_BASIC_INFO,(task_info_t)&task,&n);
    if (a!=KERN_SUCCESS || b!=KERN_SUCCESS || c!=KERN_SUCCESS) return 0;
    /* File-backed pages can be on the active queue yet remain reclaimable.
     * max() avoids adding overlapping inactive/speculative/external counts.
     * This is intentionally not a free-pages-only admission. */
    uint64_t reclaim=(uint64_t)vm.inactive_count+vm.speculative_count;
    if ((uint64_t)vm.external_page_count>reclaim) reclaim=vm.external_page_count;
    *available=((uint64_t)vm.free_count+reclaim)*page;
    *resident=(uint64_t)task.resident_size;
    task_vm_info_data_t footprint;n=TASK_VM_INFO_COUNT;
    if (task_info(mach_task_self(),TASK_VM_INFO,(task_info_t)&footprint,&n)!=KERN_SUCCESS) return 0;
    if ((uint64_t)footprint.phys_footprint>*resident) *resident=footprint.phys_footprint;
    return 1;
}
static void *ds4_ns_watch_memory(void *arg) {
    ds4_ns_memory_guard *g=(ds4_ns_memory_guard *)arg;
    while (!atomic_load(&g->stop)) {
        uint64_t available=0,resident=0;
        if (!ds4_ns_memory_sample(&available,&resident) || resident>g->budget || available<g->floor) {
            atomic_store(&g->failed,1);
            fprintf(stderr,"MEMORY_BUDGET_FAILED resident=%"PRIu64" budget=%"PRIu64
                    " available=%"PRIu64" floor=%"PRIu64"; aborting before page-file crawl\n",
                    resident,g->budget,available,g->floor);
            fflush(stderr);_Exit(3);
        }
        struct timespec delay={0,250000000};nanosleep(&delay,NULL);
    }
    return NULL;
}
static int ds4_ns_guard_begin(ds4_ns_memory_guard *g,uint64_t budget,uint64_t floor) {
    memset(g,0,sizeof(*g));g->budget=budget;g->floor=floor;
    size_t n=sizeof(g->physical);uint64_t available=0,resident=0;
    if (sysctlbyname("hw.memsize",&g->physical,&n,NULL,0) ||
        !ds4_ns_memory_sample(&available,&resident) || !floor || budget>g->physical ||
        floor>=g->physical || resident>budget || available<floor ||
        budget>available+resident) return 0;
    atomic_init(&g->stop,0);atomic_init(&g->failed,0);
    if (pthread_create(&g->thread,NULL,ds4_ns_watch_memory,g)) return 0;
    g->started=1;
    fprintf(stderr,"MEMORY_ADMITTED budget=%"PRIu64" floor=%"PRIu64" physical=%"PRIu64
            " available=%"PRIu64" resident=%"PRIu64" sample_ms=250\n",
            budget,floor,g->physical,available,resident);
    return 1;
}
static void ds4_ns_guard_end(ds4_ns_memory_guard *g) {
    if (g->started) {atomic_store(&g->stop,1);pthread_join(g->thread,NULL);g->started=0;}
}
static ds4_ns_memory_guard ds4_ns_server_memory;
static void ds4_ns_memory_stop(void){ds4_ns_guard_end(&ds4_ns_server_memory);}
static int ds4_ns_memory_start(void){
    const char*enable=ds4_nf_env("DS4_NATIVE_SESSION");if(!enable||!strcmp(enable,"0"))return 0;
    const char*budget_text=ds4_nf_env("DS4_NATIVE_SESSION_HOST_BYTES");
    const char*floor_text=ds4_nf_env("DS4_NATIVE_SESSION_HOST_FLOOR_BYTES");uint64_t budget=0,floor=0;
    if(strcmp(enable,"1")||!budget_text||!floor_text||!ds4_nf_u64(&budget_text,&budget)||*budget_text||
       !ds4_nf_u64(&floor_text,&floor)||*floor_text||floor<(1ull<<30)||
       !ds4_ns_guard_begin(&ds4_ns_server_memory,budget,floor)) {
        fprintf(stderr,"native session explicit host budget/floor refused before engine startup\n");return -1;
    }
    if(atexit(ds4_ns_memory_stop)){ds4_ns_memory_stop();return -1;}return 0;
}
