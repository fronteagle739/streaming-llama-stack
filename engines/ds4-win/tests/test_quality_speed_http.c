/* CPU-only parser/authentication gate; no model or CUDA initialization. */
#define DS4_SERVER_TEST
#define DS4_SERVER_TEST_NO_MAIN
#include "../ds4_server.c"
int main(void) {
    test_quality_speed_request_validation();
#ifdef _WIN32
    _putenv_s("DS4_SERVER_AUTH_TOKEN", "TEST-quality-secret");
#else
    setenv("DS4_SERVER_AUTH_TOKEN", "TEST-quality-secret", 1);
#endif
    const char *ok="POST /v1/quality-speed HTTP/1.1\r\nAuthorization: Bearer TEST-quality-secret\r\n\r\n";
    const char *bad="Authorization: Bearer wrong\r\n\r\n";
    const char *duplicate="Authorization: Bearer TEST-quality-secret\r\nAuthorization: Bearer TEST-quality-secret\r\n\r\n";
    TEST_ASSERT(quality_authorized(ok,strlen(ok)));
    TEST_ASSERT(!quality_authorized(bad,strlen(bad)));
    TEST_ASSERT(!quality_authorized(duplicate,strlen(duplicate)));
    TEST_ASSERT(!quality_authorized("",0));
    quality_setting setting;bool present=false;
    TEST_ASSERT(quality_config_body("{\"value\":\"max_speed\"}",true,&present,&setting));
    TEST_ASSERT(present && setting.value==1.36125f && !setting.per_phase);
    TEST_ASSERT(quality_config_body("{\"quality_speed\":{\"reasoning\":\"max_speed\",\"tool_call\":\"balanced\",\"answer\":0.35},\"messages\":[]}",false,&present,&setting));
    TEST_ASSERT(present && setting.per_phase && setting.phases[0]==1.36125f && setting.phases[1]==.35f && setting.phases[2]==.35f);
    TEST_ASSERT(quality_config_body("{\"reasoning\":\"fast\",\"answer\":\"balanced\"}",true,&present,&setting));
    TEST_ASSERT(setting.phases[0]==.65f && setting.phases[2]==.35f);
    TEST_ASSERT(!quality_config_body("{\"value\":0,\"answer\":0.35}",true,&present,&setting));
    TEST_ASSERT(!quality_config_body("{\"answer\":0,\"answer\":0.35}",true,&present,&setting));
    TEST_ASSERT(!quality_config_body("{\"value\":{\"reasoning\":1.361251}}",true,&present,&setting));
    TEST_ASSERT(!quality_config_body("{\"value\":{\"answer\":true}}",true,&present,&setting));
    TEST_ASSERT(!quality_config_body("{\"value\":{}}",true,&present,&setting));
    TEST_ASSERT(!quality_config_body("{\"value\":{\"other\":0}}",true,&present,&setting));
    TEST_ASSERT(quality_config_body("{\"value\":{\"answer\":\"faster\"}}",true,&present,&setting));
    TEST_ASSERT(setting.phases[3]==.9f);
    TEST_ASSERT(quality_config_body("{\"value\":{\"answer\":\"balanced\",\"code\":\"max_speed\"}}",true,&present,&setting));
    TEST_ASSERT(setting.phases[2]==.35f && setting.phases[3]==1.36125f);
    if (test_failures) return 1;
    puts("quality-speed HTTP range, duplicate-field, unsupported and auth gates: PASS");
    return 0;
}
