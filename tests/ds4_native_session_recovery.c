/* Mac native transport recovery: the RTX's refusal text reaches the log, the
   chat error and the notice; a failed RTX prefill falls back to the Mac; an
   invalid conversation is rebuilt with a notice.
   Build: cc -std=c11 -I engines/ds4-mac tests/ds4_native_session_recovery.c -o /tmp/t && /tmp/t
          (or cl /nologo /TC /I engines\ds4-mac tests\ds4_native_session_recovery.c) */
#include "ds4_native_session_recovery.h"
#include <stdio.h>

static int failures = 0;
static void expect(int ok, const char *what) { printf("%s %s\n", ok ? "PASS" : "FAIL", what); if (!ok) ++failures; }
static int json_safe(const char *s) { for (; *s; s++) if (*s == '"' || *s == '\\' || (unsigned char)*s < 32 || (unsigned char)*s >= 127) return 0; return 1; }

int main(void) {
    char out[DS4_NS_DETAIL_CAP], notice[DS4_NS_NOTICE_CAP];
    /* Item 4: FastAPI 409 body from the RTX broker. */
    const char *refused = "{\"detail\":\"Persistent session host admission refused: 3.6 GiB available, 9.0 GiB needed\"}";
    ds4_ns_http_detail(refused, strlen(refused), 409, out, sizeof(out));
    expect(!strcmp(out, "RTX HTTP 409: Persistent session host admission refused: 3.6 GiB available, 9.0 GiB needed"), "409 detail carried verbatim");
    /* A refusal naming a foreign GPU process: Windows path backslashes and quotes stay JSON-safe. */
    const char *gpu = "{\"detail\":\"Exclusive native admission refused: another GPU workload is running: PID 31180 \\\"D:\\\\cosmos-venv\\\\python.exe -c import flash_attn_interface\\\"\"}";
    ds4_ns_http_detail(gpu, strlen(gpu), 409, out, sizeof(out));
    expect(strstr(out, "PID 31180") && strstr(out, "cosmos-venv") && json_safe(out), "GPU holder PID and command line kept, JSON-safe");
    ds4_ns_http_detail("<html>bad gateway</html>", 24, 502, out, sizeof(out));
    expect(!strcmp(out, "RTX HTTP 502: <html>bad gateway</html>"), "non-JSON body kept raw");
    ds4_ns_http_detail(NULL, 0, 500, out, sizeof(out));
    expect(!strcmp(out, "RTX HTTP 500: (empty body)"), "empty body named");
    const char *status = "{\"session\":\"ab\",\"state\":\"failed\",\"error\":\"Persistent session GPU admission refused\"}";
    expect(ds4_ns_json_field(status, strlen(status), "error", out, sizeof(out)) && !strcmp(out, "Persistent session GPU admission refused"), "session status error field");
    char longtext[2000]; memset(longtext, 'x', sizeof(longtext) - 1); longtext[sizeof(longtext) - 1] = 0;
    ds4_ns_safe_text(out, 64, longtext, strlen(longtext));
    expect(strlen(out) == 63 && !strcmp(out + 60, "..."), "long text truncated with ellipsis");
    /* Item 2c: fallback policy and notice. */
    expect(ds4_ns_should_fallback(false, false), "refused turn falls back to Mac prefill");
    expect(!ds4_ns_should_fallback(true, false), "cancelled turn does not fall back");
    expect(!ds4_ns_should_fallback(false, true), "qualification (force_remote) never falls back");
    ds4_ns_fallback_notice(notice, sizeof(notice), "RTX failed: Exclusive native admission refused: another GPU workload is running");
    expect(!strncmp(notice, "RTX busy (RTX failed: Exclusive native admission refused", 56) && strstr(notice, "prefilling on Mac: slower") && json_safe(notice), "fallback notice text");
    ds4_ns_fallback_notice(notice, sizeof(notice), "");
    expect(!strcmp(notice, "RTX busy (no reason given), prefilling on Mac: slower"), "fallback notice without detail");
    /* Item 3: rebuild notice, combined with a fallback in the same turn. */
    ds4_ns_rebuild_notice(notice, sizeof(notice), "native decode synchronization failed");
    expect(!strcmp(notice, "Rebuilding conversation state (native decode synchronization failed): re-prefilling history"), "rebuild notice text");
    ds4_ns_notice_append(notice, sizeof(notice), "RTX busy (x), prefilling on Mac: slower");
    expect(strstr(notice, "re-prefilling history; RTX busy (x)") != NULL, "rebuild + fallback notices combined");
    printf("%s (%d failures)\n", failures ? "FAILED" : "ALL PASS", failures);
    return failures ? 1 : 0;
}
