/* SLVP v1: exact SLVI projections plus absolute spans over the existing
 * authenticated native-session channel. No image/model files on the network. */
typedef struct {unsigned char *data;size_t bytes;char sha256[65];} ds4_ns_vision_body;
static bool ds4_ns_vision_pack(const ds4_tokens*tokens,const ds4_vision_span*images,
                             size_t count,ds4_ns_vision_body*out){
    memset(out,0,sizeof(*out));if(!tokens||tokens->len<1||tokens->len>1000000||!images||!count||count>16)return false;
    uint64_t payload=0,previous=0;size_t nt=(size_t)tokens->len*4;
    for(size_t i=0;i<count;i++){
        uint64_t start=images[i].token_start,n=images[i].embedding.token_count;
        if(!n||n>32768||start<previous||start+n>=1000001||!images[i].embedding.data)return false;
        payload+=48+n*5120*4;previous=start+n;
    }
    if(payload+12+nt+4096>(64ull<<20))return false;
    unsigned char*data=malloc((size_t)payload+12+nt+4096);if(!data)return false;
    char meta[4096];ds4_ns_text b={meta,sizeof(meta),true};ds4_ns_put(&b,"[");
    size_t cursor=12+nt+4096;
    for(size_t i=0;i<count;i++){
        const ds4_vision_embedding*e=&images[i].embedding;
        uint32_t header[4]={0x49564c53,1,e->token_count,5120};size_t bytes=48+(size_t)e->token_count*5120*4;
        memcpy(data+cursor,header,16);memcpy(data+cursor+16,e->fingerprint,32);
        memcpy(data+cursor+48,e->data,bytes-48);char hash[65];ds4_native_hash(data+cursor,bytes,hash);
        ds4_ns_put(&b,"%s{\"position\":%u,\"count\":%u,\"sha256\":\"%s\"}",i?",":"",images[i].token_start,e->token_count,hash);
        cursor+=bytes;
    }
    ds4_ns_put(&b,"]");if(!b.ok){free(data);return false;}
    uint32_t nm=(uint32_t)(b.p-meta),n32=(uint32_t)nt;
    memmove(data+12+nt+nm,data+12+nt+4096,(size_t)payload);
    memcpy(data,"SLVP",4);memcpy(data+4,&n32,4);memcpy(data+8,&nm,4);
    memcpy(data+12,tokens->v,nt);memcpy(data+12+nt,meta,nm);
    out->data=data;out->bytes=12+nt+nm+(size_t)payload;ds4_native_hash(data,out->bytes,out->sha256);return true;
}
static bool ds4_ns_vision_command(ds4_ns_session*s,ds4_nf_io*io,const ds4_tokens*tokens,
        const ds4_vision_span*images,size_t count,ds4_ns_reply*r){
    ds4_ns_vision_body wire;if(!ds4_ns_vision_pack(tokens,images,count,&wire))return false;
    char extra[16384];ds4_ns_text b={extra,sizeof(extra),true};ds4_ns_put(&b,"X-DSV41-Control: ");
    ds4_ns_command_prefix(&b,s,"prefill");ds4_ns_put(&b,",\"base\":");ds4_ns_checkpoint_text(&b,&s->synced);
    ds4_ns_put(&b,"}\r\nX-DSV41-Body-SHA256: %s\r\n",wire.sha256);
    ds4_ns_reply queued;bool ok=b.ok&&ds4_ns_open(s,io,"POST","/vision","application/vnd.studio.dsv41.projected-v1",extra,wire.bytes)&&
        dsv41_c_transfer(ds4_ns_send_page,io,wire.data,wire.bytes)&&ds4_ns_body(io,202,&queued)&&
        !strcmp(queued.session,s->id)&&queued.sequence==s->sequence;
    free(wire.data);if(!ok)return false;
    char suffix[80];snprintf(suffix,sizeof(suffix),"/commands/%llu",(unsigned long long)s->sequence);
    return ds4_ns_poll(s,io,suffix,false,r);
}
