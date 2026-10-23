/* Reused from private provider e1e45374ddc307c75bd7fc0e4b8dccce6b0af6e2, tests/test_quality_request.c. */
/* Production parser -> queued request -> real runtime publication.
 * The companion test fixture enables capability without loading a model. */
#define DS4_SERVER_TEST
#define DS4_SERVER_TEST_NO_MAIN
#include "../ds4_server.c"

static ds4_quality_speed published;
extern void quality_test_enable(void);
static void queued(const char *body) {
    bool present=false;quality_setting setting;
    TEST_ASSERT(quality_config_body(body,false,&present,&setting));
    request req={0};request_quality_store(&req,present,&setting);
    /* A live change after enqueue must not mutate the stored request. */
    memset(&setting,0,sizeof(setting));
    TEST_ASSERT(request_quality_apply(&req)==0);
    ds4_quality_speed_get(&published);
}
int main(void) {
    quality_test_enable();
    const char *body="{\"quality_speed\":{\"reasoning\":\"fast\",\"tool_call\":\"balanced\",\"answer\":\"max_speed\",\"code\":\"max_quality\"},\"messages\":[]}";
    queued(body);
    TEST_ASSERT(published.revision==1 && published.per_phase);
    const float expected[4]={.65f,.35f,1.36125f,0};
    TEST_ASSERT(!memcmp(published.phase_values,expected,sizeof(expected)));
    /* Live update followed by the next saved-map request: independent Code
     * must survive that request too, rather than inherit Answer. */
    bool present=false;quality_setting live;
    TEST_ASSERT(quality_config_body("{\"answer\":\"max_speed\",\"code\":\"fast\"}",true,&present,&live));
    TEST_ASSERT(quality_apply(&live)==0);
    ds4_quality_speed_get(&published);
    TEST_ASSERT(published.phase_values[3]==.65f);
    queued(body);
    TEST_ASSERT(published.revision==3 && published.phase_values[3]==0 && published.phase_values[2]==1.36125f);
    queued("{\"quality_speed\":{\"answer\":\"balanced\"}}");
    TEST_ASSERT(published.phase_values[3]==.35f);
    queued("{\"quality_speed\":\"max_quality\"}");
    TEST_ASSERT(!published.per_phase && published.requested==0);
    const uint64_t before=published.revision;queued("{\"messages\":[]}");
    TEST_ASSERT(published.revision==before);
    const float legacy[3]={.9f,.65f,.35f};
    TEST_ASSERT(ds4_quality_speed_set_phases(legacy)==0);
    ds4_quality_speed_get(&published);
    TEST_ASSERT(published.phase_values[3]==.35f);
    const char *invalid[]={
        "{\"quality_speed\":{\"cod\":0}}", "{\"quality_speed\":{\"Code\":0}}",
        "{\"quality_speed\":{\"answer\":0,\"code\":false}}",
        "{\"quality_speed\":{\"code\":\"unknown\"}}",
        "{\"quality_speed\":{\"code\":1.361251}}",
        "{\"quality_speed\":{\"code\\u0000extra\":0}}",
        "{\"quality_speed\":{\"code\":\"max_quality\\u0000extra\"}}",
        "{\"quality_speed\\u0000extra\":0}",
        "{\"quality_speed\":{\"code\":0,\"code\":0.65}}"
    };
    for(size_t i=0;i<sizeof(invalid)/sizeof(*invalid);i++)
        TEST_ASSERT(!quality_config_body(invalid[i],false,&present,&live));
    TEST_ASSERT(!quality_config_body("{\"code\":0,\"cod\":0}",true,&present,&live));
    if(test_failures)return 1;
    puts("Native four-phase queued-request persistence, next request, inheritance and rejection: PASS");
    return 0;
}
