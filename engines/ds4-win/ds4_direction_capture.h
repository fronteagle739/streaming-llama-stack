/* Bounded teacher-forcing packet. Model geometry is returned by the engine,
 * never assumed to be the earlier 43x4096 model. No paths are accepted. */
#ifndef DS4_DIRECTION_CAPTURE_H
#define DS4_DIRECTION_CAPTURE_H
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
enum { DS4_DIRECTION_MAX_TOKENS=2048, DS4_DIRECTION_MAX_POSITIONS=2048, DS4_DIRECTION_MAX_GROUPS=16 };
typedef struct {
    uint32_t count,selected,groups,group_counts[DS4_DIRECTION_MAX_GROUPS];
    int tokens[DS4_DIRECTION_MAX_TOKENS];
    uint32_t positions[DS4_DIRECTION_MAX_POSITIONS],group[DS4_DIRECTION_MAX_POSITIONS];
    bool means;
} ds4_direction_packet;
static uint32_t ds4_direction_u32(const unsigned char *p) {
    return (uint32_t)p[0] | (uint32_t)p[1]<<8 | (uint32_t)p[2]<<16 | (uint32_t)p[3]<<24;
}
/* V1: version, token count, selected count, IDs, sorted positions (raw rows).
 * V2: version, token count, selected count, group count, IDs, (position,group)
 * pairs. V2 exports group means in one pass without retaining all token rows. */
static bool ds4_direction_capture_packet(const unsigned char *data,size_t bytes,
        uint32_t vocab,ds4_direction_packet *out) {
    if (!data || !out || !vocab || bytes<12)return false;
    const uint32_t version=ds4_direction_u32(data),n=ds4_direction_u32(data+4),k=ds4_direction_u32(data+8);
    if(version!=1 && version!=2)return false;
    const bool means=version==2;
    if(means && bytes<16)return false;
    const uint32_t header=means?4u:3u,groups=means?ds4_direction_u32(data+12):k;
    if (!n || n>DS4_DIRECTION_MAX_TOKENS || !k || k>DS4_DIRECTION_MAX_POSITIONS ||
        !groups || groups>DS4_DIRECTION_MAX_GROUPS ||
        bytes!=(size_t)(header+n+k*(means?2u:1u))*4u) return false;
    for(unsigned i=0;i<DS4_DIRECTION_MAX_GROUPS;i++)out->group_counts[i]=0;
    for(uint32_t i=0;i<n;i++) {
        uint32_t token=ds4_direction_u32(data+4u*(header+i));
        if(token>=vocab) return false;out->tokens[i]=(int)token;
    }
    for(uint32_t i=0;i<k;i++) {
        uint32_t position=ds4_direction_u32(data+4u*(header+n+i*(means?2u:1u)));
        uint32_t group=means?ds4_direction_u32(data+4u*(header+n+2u*i+1u)):i;
        if(position>=n || group>=groups || (i && position<=out->positions[i-1]))return false;
        out->positions[i]=position;out->group[i]=group;out->group_counts[group]++;
    }
    for(uint32_t i=0;i<groups;i++)if(!out->group_counts[i])return false;
    out->count=n;out->selected=k;out->groups=groups;out->means=means;return true;
}
#endif
