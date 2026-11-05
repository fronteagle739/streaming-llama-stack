/* Authenticated frozen-prefix adapter for the existing streamed LoRA recipe.
 * Input is the established exact little-endian token-ID representation.
 * Output is the native final normalized head input, F32 little-endian.
 * No model, adapter, output-path or process argument is accepted over HTTP. */
#ifndef DS4_TRAINING_API_H
#define DS4_TRAINING_API_H
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include "ds4_direction_capture.h"

static bool ds4_training_layer_path(const char *path, uint32_t layers, uint32_t *layer) {
    const char *prefix = "/v1/train/residuals/";
    if (!path || !layer || !layers) return false;
    if (strncmp(path, prefix, strlen(prefix))) prefix = "/v1/train/parity/";
    if (strncmp(path, prefix, strlen(prefix))) return false;
    const char *p = path + strlen(prefix);uint32_t value = 0;
    if (!*p) return false;
    for (; *p; ++p) {
        if (*p < '0' || *p > '9' || value > 256u) return false;
        value = value * 10u + (uint32_t)(*p - '0');
    }
    if (value >= layers) return false;
    *layer = value;return true;
}

static bool ds4_training_tokens_le32(const unsigned char *bytes, size_t size,
                                    int *tokens, uint32_t *count, uint32_t vocab) {
    if (!bytes || !tokens || !count || size == 0 || size % 4u || size > 256u * 4u || !vocab) return false;
    *count = (uint32_t)(size / 4u);
    for (uint32_t i = 0; i < *count; ++i) {
        const unsigned char *p = bytes + 4u * i;
        uint32_t value = (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
                         ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
        if (value >= vocab) return false;
        tokens[i] = (int)value;
    }
    return true;
}

#ifndef DS4_TRAINING_PARSER_ONLY
static void ds4_direction_model(server *s,int fd) {
    uint8_t identity[32];char hex[65];
    if(!ds4_engine_direction_identity(s->engine,identity)) {http_error(fd,s->enable_cors,409,"complete model/tokenizer/revision identity unavailable");return;}
    for(unsigned i=0;i<32;i++)snprintf(hex+2*i,3,"%02x",identity[i]);
    const bool single=s->slots && s->slot_count==1 && !s->batched_mode && !s->peer_decode_enabled &&
        !s->peer_prefill_enabled && ds4_session_direction_supported(s->slots[0].session);
    buf response={0};buf_printf(&response,"{\"native_id\":\"%s\",\"layers\":%d,\"hidden\":%d,\"capture\":%s,\"attach\":%s,\"method\":\"target-weight-rank-one-v1\",\"vocab\":%d}\n",hex,
        ds4_engine_output_features_layers(s->engine),ds4_engine_output_features_dim(s->engine),
        single?"true":"false",single && !s->training_features?"true":"false",ds4_engine_vocab_size(s->engine));
    http_response(fd,s->enable_cors,200,"application/json",response.ptr);buf_free(&response);
}
/* Dedicated provider only until per-request bank binding and KV-cache identity
 * are integrated. This endpoint cannot alter an owner's generation server. */
static void ds4_direction_stack(server *s,int fd,const http_request *hr) {
    const unsigned char *data=(const unsigned char *)hr->body;
    if(hr->body_len<48 || ds4_direction_u32(data)!=1 || ds4_direction_u32(data+4)>8 ||
        ds4_direction_u32(data+8)!=(uint32_t)ds4_engine_output_features_layers(s->engine) ||
        ds4_direction_u32(data+12)!=(uint32_t)ds4_engine_output_features_dim(s->engine)) {
        http_error(fd,s->enable_cors,400,"direction stack packet version/count/model geometry invalid");return;
    }
    const uint32_t count=ds4_direction_u32(data+4),layers=ds4_direction_u32(data+8),hidden=ds4_direction_u32(data+12);
    const uint64_t floats=(uint64_t)layers*2u*count*(hidden+1u);
    if(floats>8u*40u*2u*5121u || hr->body_len!=48u+floats*sizeof(float)) {
        http_error(fd,s->enable_cors,400,"bounded complete direction stack required");return;
    }
    float *pack=floats?malloc((size_t)floats*sizeof(float)):NULL;
    if(floats && !pack) {http_error(fd,s->enable_cors,500,"direction stack allocation failed");return;}
    if(floats)memcpy(pack,data+48,(size_t)floats*sizeof(float));
    char error[256]={0};int ok=0;
    pthread_mutex_lock(&s->inference_mu);
    if(s->training_features && s->slot_count==1 && !s->batched_mode && s->slots)
        ok=ds4_session_direction_stack(s->slots[0].session,data+16,pack,count,floats,error,sizeof(error));
    else snprintf(error,sizeof(error),"dedicated single-slot provider required; generation attachment not qualified");
    pthread_mutex_unlock(&s->inference_mu);free(pack);
    if(!ok) {http_error(fd,s->enable_cors,409,error);return;}
    strcpy(s->slots[0].direction_key,"!");
    http_response(fd,s->enable_cors,200,"application/json","{\"applied\":true,\"scope\":\"dedicated-session\",\"method\":\"target-weight-rank-one-v1\"}\n");
}
static void ds4_training_send_tokens(server *s, int fd, const ds4_tokens *tokens) {
    if (!tokens || tokens->len < 1 || tokens->len > 256) {
        http_error(fd, s->enable_cors, 400, "training tokens must have length 1..256");return;
    }
    unsigned char data[1024];
    for (int i = 0; i < tokens->len; ++i) {
        const uint32_t t = (uint32_t)tokens->v[i];
        for (int b = 0; b < 4; ++b) data[4*i+b] = (unsigned char)(t >> (8*b));
    }
    buf header = {0};
    buf_printf(&header, "HTTP/1.1 200 OK\r\nContent-Type: application/octet-stream\r\nContent-Length: %zu\r\nConnection: close\r\n\r\n", (size_t)tokens->len * 4);
    if (send_all(fd, header.ptr, header.len)) send_all(fd, data, (size_t)tokens->len * 4);
    buf_free(&header);
}

static void ds4_training_prompt(server *s, int fd, const http_request *hr) {
    request req; char error[256] = {0};
    if (hr->body_len > 16384 || !parse_chat_request(s->engine, s, hr->body,
            1, 256, &req, error, sizeof(error))) {
        http_error(fd, s->enable_cors, 400, error[0] ? error : "training prompt exceeds bound");return;
    }
    ds4_training_send_tokens(s, fd, &req.prompt);request_free(&req);
}

static void ds4_training_text(server *s, int fd, const http_request *hr) {
    if (!hr->body_len || hr->body_len > 16384 || strlen(hr->body) != hr->body_len) {
        http_error(fd, s->enable_cors, 400, "bounded UTF-8 answer text required");return;
    }
    ds4_tokens tokens = {0};ds4_tokenize_text(s->engine, hr->body, &tokens);
    ds4_training_send_tokens(s, fd, &tokens);ds4_tokens_free(&tokens);
}

static void ds4_direction_tokens(server *s,int fd,const http_request *hr) {
    if(!hr->body_len || hr->body_len>65536 || strlen(hr->body)!=hr->body_len) {
        http_error(fd,s->enable_cors,400,"bounded UTF-8 passage required");return;
    }
    ds4_tokens ids={0};ds4_tokenize_text(s->engine,hr->body,&ids);
    if(ids.len<1 || ids.len>DS4_DIRECTION_MAX_TOKENS) {
        ds4_tokens_free(&ids);http_error(fd,s->enable_cors,400,"passage must tokenize to1..2048 tokens");return;
    }
    buf result={0};buf_printf(&result,"{\"tokens\":[");
    for(int i=0;i<ids.len;i++)buf_printf(&result,"%s%d",i?",":"",ids.v[i]);
    buf_printf(&result,"],\"byte_offsets\":[");
    size_t cursor=0;bool ok=true;
    for(int i=0;i<ids.len;i++) {
        size_t length=0;char *piece=ds4_token_text(s->engine,ids.v[i],&length);
        if(!piece || length>hr->body_len-cursor || memcmp(hr->body+cursor,piece,length))ok=false;
        if(ok)buf_printf(&result,"%s[%zu,%zu]",i?",":"",cursor,cursor+length);
        free(piece);if(!ok)break;cursor+=length;
    }
    if(cursor!=hr->body_len)ok=false;
    ds4_tokens_free(&ids);
    if(!ok) {buf_free(&result);http_error(fd,s->enable_cors,409,"token pieces do not reconstruct the passage; offsets refused");return;}
    buf_printf(&result,"],\"text_bytes\":%zu}\n",cursor);
    http_response(fd,s->enable_cors,200,"application/json",result.ptr);buf_free(&result);
}

static void ds4_direction_prompt(server *s,int fd,const http_request *hr) {
    request req;char error[256]={0};
    if(hr->body_len>65536 || !parse_chat_request(s->engine,s,hr->body,1,DS4_DIRECTION_MAX_TOKENS,&req,error,sizeof(error))) {
        http_error(fd,s->enable_cors,400,error[0]?error:"bounded direction prompt required");return;
    }
    if(req.prompt.len<1 || req.prompt.len>DS4_DIRECTION_MAX_TOKENS) {
        request_free(&req);http_error(fd,s->enable_cors,400,"direction prompt exceeds2048 tokens");return;
    }
    buf result={0};buf_printf(&result,"{\"tokens\":[");
    for(int i=0;i<req.prompt.len;i++)buf_printf(&result,"%s%d",i?",":"",req.prompt.v[i]);
    buf_printf(&result,"]}\n");request_free(&req);
    http_response(fd,s->enable_cors,200,"application/json",result.ptr);buf_free(&result);
}

static void ds4_training_directions_selected(server *s,int fd,const http_request *hr,const direction_selection *selection) {
    ds4_direction_packet packet;
    const bool parity=!strcmp(hr->path,"/v1/train/direction-parity");
    const int dim=ds4_engine_direction_features_dim(s->engine,parity);
    if(!dim || !ds4_direction_capture_packet((const unsigned char *)hr->body,hr->body_len,
            (uint32_t)ds4_engine_vocab_size(s->engine),&packet)) {
        http_error(fd,s->enable_cors,400,"versioned bounded exact-token/selected-position packet required");return;
    }
    const size_t row_bytes=(size_t)dim*sizeof(float),bytes=(size_t)packet.groups*row_bytes;
    if(bytes+(packet.means?row_bytes:0)>128u*1024u*1024u) {http_error(fd,s->enable_cors,400,"direction capture exceeds128MiB bound");return;}
    float *features=packet.means?calloc((size_t)packet.groups*dim,sizeof(float)):malloc(bytes);
    float *row=packet.means?malloc(row_bytes):NULL;char error[256]={0};int rc=0;
    if(!features || (packet.means && !row)) {free(features);free(row);http_error(fd,s->enable_cors,500,"direction capture allocation failed");return;}
    pthread_mutex_lock(&s->direction_mu);
    if(selection && (s->slot_count!=1 || !s->slots ||
            !direction_bind_request(s,&s->slots[0],selection,error,sizeof(error)))) {
        pthread_mutex_unlock(&s->direction_mu);free(features);free(row);
        http_error(fd,s->enable_cors,409,error[0]?error:"single native slot required");return;
    }
    pthread_mutex_lock(&s->inference_mu);
    ds4_session *session=s->slot_count==1 && !s->batched_mode && !s->peer_decode_enabled &&
        !s->peer_prefill_enabled && s->slots?s->slots[0].session:NULL;
    if(!session)snprintf(error,sizeof(error),"single local native session required for direction capture");
    else {
        uint32_t slot=0;rc=1;
        if(!s->training_features && !selection) {
            uint8_t identity[32];
            rc=ds4_engine_direction_identity(s->engine,identity) &&
                ds4_session_direction_stack(session,identity,NULL,0,0,error,sizeof(error));
            if(rc) {
                /* The following request must rebind even when its selection
                 * matches the selection before this baseline-only capture. */
                strcpy(s->slots[0].direction_key,"!");s->slots[0].direction_count=0;
                request_live_state_clear(s,&s->slots[0]);
            }
        }
        for(uint32_t i=0;rc && i<packet.count;i++) {
            if(client_socket_disconnected(fd)) {snprintf(error,sizeof(error),"direction capture cancelled by its client");rc=0;break;}
            bool keep=slot<packet.selected && packet.positions[slot]==i;
            float *dest=keep?(packet.means?row:features+(size_t)slot*dim):NULL;
            rc=ds4_session_capture_directions(session,packet.tokens[i],i==0,parity,dest,keep?dim:0,error,sizeof(error))==dim;
            if(keep) {
                if(rc && packet.means)for(int j=0;j<dim;j++)features[(size_t)packet.group[slot]*dim+j]+=row[j];
                slot++;
            }
        }
        ds4_session_invalidate(session);
        if(!s->training_features || selection) {
            strcpy(s->slots[0].direction_key,"!");s->slots[0].direction_count=0;
            request_live_state_clear(s,&s->slots[0]);
        }
    }
    pthread_mutex_unlock(&s->inference_mu);
    pthread_mutex_unlock(&s->direction_mu);
    free(row);
    if(!rc) {free(features);http_error(fd,s->enable_cors,500,error);return;}
    if(packet.means)for(uint32_t group=0;group<packet.groups;group++)for(int j=0;j<dim;j++)
        features[(size_t)group*dim+j]/=(float)packet.group_counts[group];
    buf header={0};
    buf_printf(&header,"HTTP/1.1 200 OK\r\nContent-Type: application/octet-stream\r\nContent-Length: %zu\r\nX-DS4-Feature-Kind: %s\r\nX-DS4-Layers: %d\r\nX-DS4-Hidden: %d\r\nX-DS4-HC-Streams: 4\r\nX-DS4-Selected-Count: %u\r\nX-DS4-Group-Count: %u\r\nX-DS4-Aggregation: %s\r\nX-DS4-Floats-Per-Position: %d\r\nConnection: close\r\n\r\n",
        bytes,parity?"writer-hc-logits-v1":"writer-hc-v1",ds4_engine_output_features_layers(s->engine),ds4_engine_output_features_dim(s->engine),packet.selected,packet.groups,packet.means?"mean":"raw",dim);
    if(send_all(fd,header.ptr,header.len))send_all(fd,features,bytes);
    fprintf(stderr,"DIRECTION_CAPTURE tokens=%u selected=%u groups=%u floats_per_position=%d bytes=%zu base_read_only=1\n",packet.count,packet.selected,packet.groups,dim,bytes);
    fflush(stderr);buf_free(&header);free(features);
}

static void ds4_training_directions(server *s,int fd,const http_request *hr) {
    ds4_training_directions_selected(s,fd,hr,NULL);
}

/* Bounded request-scoped parity: LE32 JSON byte count, selection JSON, existing
 * selected-position packet. All arms use the exact same supplied token IDs. */
static void ds4_direction_evaluate(server *s,int fd,const http_request *hr) {
    const unsigned char *data=(const unsigned char *)hr->body;
    if(hr->body_len<4) {http_error(fd,s->enable_cors,400,"bounded parity selection required");return;}
    const uint32_t bytes=ds4_direction_u32(data);
    if(!bytes || bytes>8192 || hr->body_len<(size_t)bytes+4u+12u || memchr(data+4,0,bytes)) {
        http_error(fd,s->enable_cors,400,"bounded parity selection required");return;
    }
    char *json=xstrndup((const char *)data+4,bytes);direction_selection selection={0};
    const bool valid=direction_parse_request(json,&selection) && selection.present;free(json);
    if(!valid) {http_error(fd,s->enable_cors,400,"valid direction_adapters selection required");return;}
    http_request capture=*hr;capture.body=hr->body+4+bytes;capture.body_len=hr->body_len-4-bytes;
    strcpy(capture.path,"/v1/train/direction-parity");
    ds4_training_directions_selected(s,fd,&capture,&selection);
}

static void ds4_training_features(server *s, int fd, const http_request *hr) {
    int tokens[256]; uint32_t count = 0;
    const int dim = ds4_engine_output_features_dim(s->engine);
    if (!dim || !ds4_training_tokens_le32((const unsigned char *)hr->body, hr->body_len,
                                         tokens, &count, (uint32_t)ds4_engine_vocab_size(s->engine))) {
        http_error(fd, s->enable_cors, 400, "original DSV4.1 and 1..256 exact LE32 token IDs required");
        return;
    }
    const size_t bytes = (size_t)count * (size_t)dim * sizeof(float);
    float *features = malloc(bytes);
    ds4_session *session = NULL;
    char error[256] = {0}; int rc = 1;
    if (!features) {http_error(fd, s->enable_cors, 500, "bounded feature buffer allocation failed");return;}
    pthread_mutex_lock(&s->inference_mu);
    if (!s->training_features || s->slot_count != 1 || s->batched_mode ||
        !s->slots || !(session = s->slots[0].session)) {
        snprintf(error, sizeof(error), "dedicated single-slot --training-features provider required");
    } else {
        ds4_tokens first = {0}; ds4_tokens_push(&first, tokens[0]);
        rc = ds4_session_sync(session, &first, error, sizeof(error));
        ds4_tokens_free(&first);
        for (uint32_t i = 0; !rc && i < count; ++i) {
            if (i) rc = ds4_session_eval(session, tokens[i], error, sizeof(error));
            if (!rc && ds4_session_copy_output_features(session, features + (size_t)i * dim, dim) != dim) {
                snprintf(error, sizeof(error), "native output feature readback failed");rc = 1;
            }
        }
    }
    pthread_mutex_unlock(&s->inference_mu);
    if (rc) {free(features);http_error(fd, s->enable_cors, 500, error);return;}
    buf header = {0};
    buf_printf(&header, "HTTP/1.1 200 OK\r\nContent-Type: application/octet-stream\r\nContent-Length: %zu\r\nX-Slstack-Hidden-Dim: %d\r\nX-Slstack-Token-Count: %u\r\nConnection: close\r\n\r\n", bytes, dim, count);
    if (send_all(fd, header.ptr, header.len)) send_all(fd, features, bytes);
    fprintf(stderr, "TRAINING_FEATURES native_frozen_prefix=1 tokens=%u hidden=%d bytes=%zu base_read_only=1\n", count, dim, bytes);
    fflush(stderr);buf_free(&header);free(features);
}
#endif
#endif
