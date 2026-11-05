/* Included after the GGUF parser: quant-independent metadata/shape digest.
 * Hashes no tensor payload. Match Studio direction_identity.read native_id. */
#ifndef DS4_DIRECTION_IDENTITY_H
#define DS4_DIRECTION_IDENTITY_H
#include "ds4_sha256.h"
typedef struct {unsigned char *key;size_t length;uint8_t hash[32];} ds4_direction_identity_record;
static int ds4_direction_identity_compare(const void *a,const void *b) {
    const ds4_direction_identity_record *x=a,*y=b;
    const size_t n=x->length<y->length?x->length:y->length;
    int order=memcmp(x->key,y->key,n);
    return order?order:x->length<y->length?-1:x->length>y->length?1:0;
}
static void ds4_direction_identity_u32(unsigned char *out,uint32_t v) {
    for(unsigned i=0;i<4;i++)out[i]=(unsigned char)(v>>(8*i));
}
static bool ds4_direction_model_identity(const ds4_model *m,uint8_t out[32]) {
    if(!m || !out || m->n_kv>100000 || m->n_tensors>200000)return false;
    const char *required[]={"general.architecture","general.name","general.source.url","general.source.revision"};
    for(unsigned i=0;i<4;i++) {ds4_str value;if(!model_get_string(m,required[i],&value) || !value.len)return false;}
    const size_t cap=(size_t)(m->n_kv+m->n_tensors);
    ds4_direction_identity_record *records=calloc(cap,sizeof(*records));
    unsigned char *joined=NULL;size_t count=0,total=sizeof("direction-model-native-v1"),tokenizer_keys=0;
    bool ok=records!=NULL;
    for(uint64_t i=0;ok && i<m->n_kv;i++) {
        const ds4_kv *kv=&m->kv[i];
        bool selected=kv->key.len>=10 && !memcmp(kv->key.ptr,"tokenizer.",10);
        if(selected)tokenizer_keys++;
        for(unsigned k=0;k<4;k++)if(ds4_streq(kv->key,required[k]))selected=true;
        if(!selected)continue;
        ds4_cursor cursor=cursor_at(m,kv->value_pos);
        if(!skip_value(&cursor,kv->type,0) || cursor.pos<kv->value_pos || cursor.pos-kv->value_pos>16u*1024u*1024u) {ok=false;break;}
        const size_t raw=(size_t)(cursor.pos-kv->value_pos);
        unsigned char *value=malloc(raw+4);
        ds4_direction_identity_record *record=&records[count++];
        record->length=(size_t)kv->key.len+1;record->key=malloc(record->length);
        if(!value || !record->key) {free(value);ok=false;break;}
        record->key[0]='K';memcpy(record->key+1,kv->key.ptr,(size_t)kv->key.len);
        ds4_direction_identity_u32(value,kv->type);memcpy(value+4,m->map+kv->value_pos,raw);
        ds4_sha256_bytes(value,raw+4,record->hash);free(value);
        total+=4+record->length+32;
    }
    for(uint64_t i=0;ok && i<m->n_tensors;i++) {
        const ds4_tensor *tensor=&m->tensors[i];
        if(tensor->ndim>DS4_MAX_DIMS) {ok=false;break;}
        ds4_direction_identity_record *record=&records[count++];
        record->length=(size_t)tensor->name.len+1;record->key=malloc(record->length);
        if(!record->key) {ok=false;break;}
        record->key[0]='T';memcpy(record->key+1,tensor->name.ptr,(size_t)tensor->name.len);
        unsigned char shape[4+8*DS4_MAX_DIMS];ds4_direction_identity_u32(shape,tensor->ndim);
        for(unsigned d=0;d<tensor->ndim;d++)for(unsigned b=0;b<8;b++)shape[4+8*d+b]=(unsigned char)(tensor->dim[d]>>(8*b));
        ds4_sha256_bytes(shape,4+8*tensor->ndim,record->hash);total+=4+record->length+32;
    }
    ok=ok && tokenizer_keys && total<=32u*1024u*1024u;
    if(ok) {
        joined=malloc(total);ok=joined!=NULL;
        if(ok) {
            qsort(records,count,sizeof(*records),ds4_direction_identity_compare);
            memcpy(joined,"direction-model-native-v1",sizeof("direction-model-native-v1"));
            size_t offset=sizeof("direction-model-native-v1");
            for(size_t i=0;i<count;i++) {
                ds4_direction_identity_u32(joined+offset,(uint32_t)records[i].length);offset+=4;
                memcpy(joined+offset,records[i].key,records[i].length);offset+=records[i].length;
                memcpy(joined+offset,records[i].hash,32);offset+=32;
            }
            ds4_sha256_bytes(joined,total,out);
        }
    }
    for(size_t i=0;i<count;i++)free(records[i].key);
    free(records);free(joined);return ok;
}
#endif
