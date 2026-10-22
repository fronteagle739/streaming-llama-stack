#ifndef DS4_QUALITY_CODE_H
#define DS4_QUALITY_CODE_H
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
/* Streaming over the actual decoded token bytes: no guessed vocabulary IDs.
 * Fences start at a line head (up to3 spaces), close on a matching-or-longer
 * backtick-only line. Delimiters can span token boundaries. No allocation,
 * text history, language guessing or model-specific vocabulary IDs. */
typedef struct {
    bool fenced, opening, closing, line_text, run_head;
    uint32_t ticks, fence_ticks, spaces;
} ds4_quality_code_tracker;
static inline bool ds4_quality_code_space(char c) {
    return c==' ' || c=='\t' || c=='\r' || c=='\n';
}
static inline void ds4_quality_code_feed(ds4_quality_code_tracker *s,const char *text,size_t n) {
    for(size_t i=0;i<n;i++) {
        const char c=text[i];
        if(c=='`') {
            if(!s->ticks)s->run_head=!s->line_text && s->spaces<=3;
            if(s->ticks<UINT32_MAX)s->ticks++;
            if(!s->fenced && s->run_head && s->ticks>=3) {
                s->fenced=true;s->opening=true;s->fence_ticks=s->ticks;
            } else if(s->fenced && s->opening)s->fence_ticks=s->ticks;
            s->line_text=true;continue;
        }
        if(s->ticks) {
            if(s->fenced) {
                if(s->opening)s->opening=false;
                else if(s->run_head && s->ticks>=s->fence_ticks && ds4_quality_code_space(c))s->closing=true;
            }
            s->ticks=0;
        }
        if(s->closing) {
            if(c=='\n'){s->fenced=false;s->closing=false;s->fence_ticks=0;}
            else if(!ds4_quality_code_space(c))s->closing=false;
        }
        if(c=='\n') {
            s->line_text=false;s->spaces=0;
        } else if(!s->line_text) {
            if(c==' '){if(s->spaces<UINT32_MAX)s->spaces++;}
            else if(c!='\r')s->line_text=true;
        }
    }
}
static inline bool ds4_quality_code_current(const ds4_quality_code_tracker *s) {
    return s->fenced;
}
#endif
