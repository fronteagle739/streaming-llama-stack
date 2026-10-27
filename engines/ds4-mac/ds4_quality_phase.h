#ifndef DS4_QUALITY_PHASE_H
#define DS4_QUALITY_PHASE_H
#include <stdbool.h>
#include <stddef.h>
#include <string.h>
#include "ds4_quality_code.h"
/* Shared with the actual 1017 chat renderer, not inferred tag spellings. */
#define DS41_TOOL_CALLS_START "<｜DSML｜ calls>"
#define DS41_TOOL_CALLS_END "</｜DSML｜ calls>"
enum {DS4_QUALITY_REASONING, DS4_QUALITY_TOOL_CALL, DS4_QUALITY_ANSWER, DS4_QUALITY_CODE, DS4_QUALITY_PHASE_COUNT};
static const char *const ds4_quality_phase_names[] = {"reasoning", "tool_call", "answer", "code"};
typedef struct {
    bool reasoning, tool;
    unsigned marker; /* 1=open, 2=close; only armed by the actual DSML token ID */
    size_t matched;
    char tail[2];
    size_t tail_len;
    ds4_quality_code_tracker code;
} ds4_quality_phase_tracker;
/* Reasoning boundaries are vocabulary IDs. DSML is a special token surrounded
 * by BPE text: require that ID plus the exact template suffix. Text spelling a
 * reasoning/DSML token without its special ID cannot trigger a switch. */
static inline void ds4_quality_phase_feed(ds4_quality_phase_tracker *s, int token,
        int think_open, int think_close, int dsml, const char *text, size_t n) {
    if (token == think_open) {s->reasoning=true;s->marker=0;s->tail_len=0;memset(&s->code,0,sizeof(s->code));return;}
    if (token == think_close) {s->reasoning=false;s->marker=0;s->tail_len=0;memset(&s->code,0,sizeof(s->code));return;}
    if (token == dsml) {
        s->marker=s->reasoning ? 0u : s->tail_len==2 && !memcmp(s->tail,"</",2) ? 2u :
            s->tail_len && s->tail[s->tail_len-1]=='<' ? 1u : 0u;
        s->matched=0;s->tail_len=0;return;
    }
    /* The suffix is taken from the shared renderer constants after DSML. */
    const char *suffix=strstr(DS41_TOOL_CALLS_START," calls>");
    for (size_t i=0;i<n;i++) {
        if (s->marker) {
            if (text[i]!=suffix[s->matched]) {s->marker=0;s->matched=0;}
            else if (!suffix[++s->matched]) {
                s->tool=s->marker==1;s->marker=0;s->matched=0;memset(&s->code,0,sizeof(s->code));
            }
        }
        if (s->tail_len==2) {s->tail[0]=s->tail[1];s->tail_len=1;}
        s->tail[s->tail_len++]=text[i];
        if(!s->tool)ds4_quality_code_feed(&s->code,text+i,1);
    }
}
static inline int ds4_quality_phase_base(const ds4_quality_phase_tracker *s) {
    return s->reasoning ? DS4_QUALITY_REASONING : s->tool ? DS4_QUALITY_TOOL_CALL : DS4_QUALITY_ANSWER;
}
static inline int ds4_quality_phase_current(const ds4_quality_phase_tracker *s) {
    const int base=ds4_quality_phase_base(s);
    return base!=DS4_QUALITY_TOOL_CALL && ds4_quality_code_current(&s->code) ? DS4_QUALITY_CODE : base;
}
static inline float ds4_quality_phase_ramp(float from,float to,unsigned step,unsigned ramp) {
    return !ramp || step>=ramp ? to : from+(to-from)*(float)step/(float)ramp;
}
#endif
