/* Immutable process-lifetime vector bank and request-scoped native attachment.
 * No model or adapter path is accepted over HTTP. Header included after server,
 * JSON, HTTP, live-state and cache helpers, before generate_job_inner. */
#ifndef DS4_DIRECTION_SERVER_H
#define DS4_DIRECTION_SERVER_H
#include "ds4_sha256.h"
#include "ds4_direction_capture.h"

static void direction_hex(char out[65],const uint8_t bytes[32]) {
    for(unsigned i=0;i<32;i++)snprintf(out+2*i,3,"%02x",bytes[i]);
}

static bool direction_valid_id(const char *id) {
    if(!id || strlen(id)!=64)return false;
    for(unsigned i=0;i<64;i++)if(!((id[i]>='0' && id[i]<='9') || (id[i]>='a' && id[i]<='f')))return false;
    return true;
}

static bool direction_parse_selection(const char **p,direction_selection *out) {
    json_ws(p);if(*(*p)++!='[')return false;
    json_ws(p);
    while(**p!=']') {
        if(out->count>=DIRECTION_STACK_CAPACITY || *(*p)++!='{')return false;
        direction_selection_row row={0};bool id_seen=false,strength_seen=false;
        json_ws(p);
        while(**p!='}') {
            char *key=NULL;if(!json_string(p,&key))return false;
            json_ws(p);if(*(*p)++!=':') {free(key);return false;}
            bool ok=false;
            if(!strcmp(key,"id") && !id_seen) {
                char *value=NULL;ok=json_string(p,&value) && direction_valid_id(value);
                if(ok)memcpy(row.id,value,65);free(value);id_seen=true;
            } else if(!strcmp(key,"strength") && !strength_seen) {
                double value=0;ok=json_number(p,&value) && isfinite(value) && value>=0 && value<=4;
                if(ok)row.strength=(float)value;strength_seen=true;
            }
            free(key);if(!ok)return false;
            json_ws(p);
            if(**p=='}')break;
            if(*(*p)++!=',')return false;
            json_ws(p);if(**p=='}')return false;
        }
        if(!id_seen || !strength_seen || *(*p)++!='}')return false;
        for(uint32_t i=0;i<out->count;i++)if(!strcmp(out->row[i].id,row.id))return false;
        out->row[out->count++]=row;
        json_ws(p);if(**p==']')break;
        if(*(*p)++!=',')return false;
        json_ws(p);if(**p==']')return false;
    }
    (*p)++;
    /* Stable additive order and KV identity regardless of GUI list order. */
    for(uint32_t i=1;i<out->count;i++)for(uint32_t j=i;j && strcmp(out->row[j-1].id,out->row[j].id)>0;j--) {
        direction_selection_row tmp=out->row[j];out->row[j]=out->row[j-1];out->row[j-1]=tmp;
    }
    return true;
}

static bool direction_parse_request(const char *body,direction_selection *out) {
    if(!body || !out)return false;
    memset(out,0,sizeof(*out));const char *p=body;json_ws(&p);if(*p++!='{')return false;
    json_ws(&p);
    while(*p && *p!='}') {
        char *key=NULL;if(!json_string(&p,&key))return false;
        json_ws(&p);if(*p++!=':') {free(key);return false;}
        bool ok;
        if(!strcmp(key,"direction_adapters")) {
            ok=!out->present && direction_parse_selection(&p,out);out->present=true;
        } else ok=json_skip_value(&p);
        free(key);if(!ok)return false;
        json_ws(&p);if(*p=='}')break;
        if(*p++!=',')return false;
        json_ws(&p);if(*p=='}')return false;
    }
    if(*p++!='}')return false;
    json_ws(&p);return !*p;
}

static void direction_register(server *s,int fd,const http_request *hr) {
    const unsigned char *data=(const unsigned char *)hr->body;
    const uint32_t layers=(uint32_t)ds4_engine_output_features_layers(s->engine);
    const uint32_t hidden=(uint32_t)ds4_engine_output_features_dim(s->engine);
    const uint64_t floats=(uint64_t)layers*2u*(hidden+1u);
    uint8_t actual[32],hash[32];char id[65];
    if(layers!=40 || hidden!=5120 || hr->body_len!=48u+floats*sizeof(float) ||
        ds4_direction_u32(data)!=1 || ds4_direction_u32(data+4)!=1 ||
        ds4_direction_u32(data+8)!=layers || ds4_direction_u32(data+12)!=hidden ||
        !ds4_engine_direction_identity(s->engine,actual) || memcmp(actual,data+16,32)) {
        http_error(fd,s->enable_cors,409,"one complete direction for this model identity required");return;
    }
    float *pack=malloc((size_t)floats*sizeof(float));
    if(!pack) {http_error(fd,s->enable_cors,500,"direction bank allocation failed");return;}
    memcpy(pack,data+48,(size_t)floats*sizeof(float));bool signal=false;
    for(uint64_t offset=0;offset<floats;offset+=hidden+1u) {
        double norm=0;bool finite=true;
        for(uint32_t i=0;i<hidden;i++) {float v=pack[offset+1u+i];finite=finite && isfinite(v);norm+=(double)v*v;}
        if(pack[offset]!=1.0f || !finite || (norm>1e-12 && fabs(norm-1.0)>1e-3)) {
            free(pack);http_error(fd,s->enable_cors,400,"unit or zero vectors and unscaled registration required");return;
        }
        signal=signal || norm>1e-12;
    }
    if(!signal) {free(pack);http_error(fd,s->enable_cors,400,"direction has no signal");return;}
    ds4_sha256_bytes(data,hr->body_len,hash);direction_hex(id,hash);
    pthread_mutex_lock(&s->inference_mu);
    bool existing=false;for(uint32_t i=0;i<s->direction_bank_count;i++)existing=existing || !strcmp(id,s->direction_bank[i].id);
    if(!existing && s->direction_bank_count<DIRECTION_BANK_CAPACITY) {
        direction_bank_entry *entry=&s->direction_bank[s->direction_bank_count++];
        memcpy(entry->id,id,65);entry->pack=pack;pack=NULL;
    }
    pthread_mutex_unlock(&s->inference_mu);
    if(pack && !existing) {free(pack);http_error(fd,s->enable_cors,409,"direction bank is full; restart the engine before registering more");return;}
    free(pack);buf response={0};buf_printf(&response,"{\"registered\":true,\"id\":\"%s\"}\n",id);
    http_response(fd,s->enable_cors,200,"application/json",response.ptr);buf_free(&response);
}

static bool direction_bind_request(server *s,server_slot *slot,const direction_selection *selection,char *err,size_t errlen) {
    uint32_t count=0;
    for(uint32_t i=0;i<selection->count;i++)if(selection->row[i].strength>0)count++;
    if(!count && !slot->direction_count && !slot->direction_key[0])return true;
    if(s->slot_count!=1 || s->batched_mode || s->peer_decode_enabled || s->peer_prefill_enabled) {
        snprintf(err,errlen,"directions require a single local native session; peer/batched attachment is unavailable");return false;
    }
    uint8_t identity[32],hash[32];char key[65]={0};
    if(!ds4_engine_direction_identity(s->engine,identity)) {snprintf(err,errlen,"model identity unavailable");return false;}
    if(count) {
        peer_sha256 sha;peer_sha256_init(&sha);peer_sha256_update(&sha,identity,32);
        for(uint32_t i=0;i<selection->count;i++)if(selection->row[i].strength>0) {
            peer_sha256_update(&sha,selection->row[i].id,64);
            peer_sha256_update(&sha,&selection->row[i].strength,sizeof(float));
        }
        peer_sha256_final(&sha,hash);direction_hex(key,hash);
    }
    if(!strcmp(key,slot->direction_key))return true;
    const uint32_t layers=40,hidden=5120;
    const uint64_t floats=(uint64_t)layers*2u*count*(hidden+1u);
    float *pack=count?malloc((size_t)floats*sizeof(float)):NULL;
    if(count && !pack) {snprintf(err,errlen,"direction stack allocation failed");return false;}
    pthread_mutex_lock(&s->inference_mu);
    uint32_t k=0;bool found=true;
    for(uint32_t i=0;i<selection->count && found;i++)if(selection->row[i].strength>0) {
        direction_bank_entry *entry=NULL;
        for(uint32_t j=0;j<s->direction_bank_count;j++)if(!strcmp(s->direction_bank[j].id,selection->row[i].id))entry=&s->direction_bank[j];
        if(!entry) {found=false;break;}
        for(uint32_t site=0;site<layers*2u;site++) {
            const uint64_t offset=((uint64_t)site*count+k)*(hidden+1u);
            pack[offset]=selection->row[i].strength;
            memcpy(pack+offset+1u,entry->pack+(uint64_t)site*(hidden+1u)+1u,hidden*sizeof(float));
        }
        k++;
    }
    if(!found)snprintf(err,errlen,"direction ID is not registered on this engine; re-register the adapter");
    const bool ok=found && ds4_session_direction_stack(slot->session,identity,pack,count,floats,err,errlen);
    if(ok) {memcpy(slot->direction_key,key,65);slot->direction_count=count;}
    pthread_mutex_unlock(&s->inference_mu);free(pack);
    if(ok) {
        request_live_state_clear(s,slot);slot->continued_last_store_tokens=0;
        fprintf(stderr,"DIRECTION_BIND slot=%d adapters=%u key=%s kv_invalidated=1 disk_cache=%s\n",
            slot->id,count,key,count?"bypassed-unscoped":"base");
    }
    return ok;
}
#endif
