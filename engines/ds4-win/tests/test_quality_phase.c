#include "../ds4_quality_phase.h"
#include <assert.h>
#include <stdio.h>
static void feed(ds4_quality_phase_tracker *s,int id,const char *text) {
    ds4_quality_phase_feed(s,id,10,11,12,text,strlen(text));
}
int main(void) {
    ds4_quality_phase_tracker s={0};
    feed(&s,1,"<think>");assert(ds4_quality_phase_current(&s)==DS4_QUALITY_ANSWER);
    feed(&s,10,"<think>");assert(ds4_quality_phase_current(&s)==DS4_QUALITY_REASONING);
    feed(&s,11,"</think>");assert(ds4_quality_phase_current(&s)==DS4_QUALITY_ANSWER);
    feed(&s,1,DS41_TOOL_CALLS_START);assert(!s.tool); /* text without DSML ID */
    feed(&s,1,"\n\n<");feed(&s,12,"｜DSML｜");feed(&s,1," cal");assert(!s.tool);
    feed(&s,1,"ls>\n");assert(ds4_quality_phase_current(&s)==DS4_QUALITY_TOOL_CALL);
    feed(&s,1,"<");feed(&s,1,"/");feed(&s,12,"｜DSML｜");feed(&s,1," calls>");assert(!s.tool);
    feed(&s,1,"<");feed(&s,12,"｜DSML｜");feed(&s,1," other>");assert(!s.tool);
    for(size_t split=0;split<=7;split++) {
        ds4_quality_phase_tracker code={0};const char *open="```cpp\n";
        ds4_quality_phase_feed(&code,1,10,11,12,open,split);
        ds4_quality_phase_feed(&code,1,10,11,12,open+split,7-split);
        assert(ds4_quality_phase_current(&code)==DS4_QUALITY_CODE);
        feed(&code,1,"int x = 1;\n``");assert(ds4_quality_phase_current(&code)==DS4_QUALITY_CODE);
        feed(&code,1,"`\nprose");assert(ds4_quality_phase_current(&code)==DS4_QUALITY_ANSWER);
    }
    ds4_quality_phase_tracker code={0};feed(&code,10,"");feed(&code,1,"```cpp\n");
    assert(ds4_quality_phase_current(&code)==DS4_QUALITY_CODE);
    assert(ds4_quality_phase_base(&code)==DS4_QUALITY_REASONING);
    feed(&code,11,"");assert(ds4_quality_phase_current(&code)==DS4_QUALITY_ANSWER);
    assert(ds4_quality_phase_ramp(1.36125f,.35f,8,8)==.35f);
    float mid=ds4_quality_phase_ramp(1.36125f,.35f,4,8);assert(mid>.35f && mid<1.36125f);
    puts("Native exact special-token phases, split DSML markers, and ramp: PASS");return 0;
}
