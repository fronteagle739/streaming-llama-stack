#include "../ds4_direction_capture.h"
#include <assert.h>
#include <stdio.h>
static void put(unsigned char *p,uint32_t x) {for(unsigned i=0;i<4;i++)p[i]=(unsigned char)(x>>(8*i));}
int main(void) {
    unsigned char data[36]={0};uint32_t fields[]={1,4,2,10,11,12,13,0,3};
    for(unsigned i=0;i<9;i++)put(data+4*i,fields[i]);
    ds4_direction_packet packet;
    assert(ds4_direction_capture_packet(data,sizeof(data),100,&packet));
    assert(packet.count==4 && packet.selected==2 && packet.groups==2 && !packet.means && packet.positions[1]==3 && packet.tokens[2]==12);
    assert(!ds4_direction_capture_packet(data,sizeof(data)-1,100,&packet));
    assert(!ds4_direction_capture_packet(data,sizeof(data),12,&packet));
    put(data+32,0);assert(!ds4_direction_capture_packet(data,sizeof(data),100,&packet));
    put(data+32,4);assert(!ds4_direction_capture_packet(data,sizeof(data),100,&packet));
    put(data+32,3);put(data+28,3);assert(!ds4_direction_capture_packet(data,sizeof(data),100,&packet));
    put(data+28,0);put(data,3);assert(!ds4_direction_capture_packet(data,sizeof(data),100,&packet));
    unsigned char grouped[64]={0};uint32_t fields2[]={2,4,4,2,10,11,12,13,0,0,1,1,2,0,3,1};
    for(unsigned i=0;i<16;i++)put(grouped+4*i,fields2[i]);
    assert(ds4_direction_capture_packet(grouped,sizeof(grouped),100,&packet));
    assert(packet.means && packet.groups==2 && packet.group_counts[0]==2 && packet.group_counts[1]==2);
    put(grouped+60,2);assert(!ds4_direction_capture_packet(grouped,sizeof(grouped),100,&packet));
    put(grouped+60,0);put(grouped+44,0);assert(!ds4_direction_capture_packet(grouped,sizeof(grouped),100,&packet));
    puts("Direction capture packet: geometry/order/bounds PASS");return 0;
}
