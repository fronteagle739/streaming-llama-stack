/* Server-only activation. The legacy mode1 qualification is left unchanged. */
/* Append-only KV owns the exact generated thinking prefix across turns. Only
 * a successfully qualified native-session server changes history rendering. */
static bool ds4_ns_preserve_history_reasoning;
static void ds4_ns_config_free(ds4_ns_config*c){if(c){ds4_ns_preserve_history_reasoning=false;memset(c->http.token,0,sizeof(c->http.token));free(c);}}
static int ds4_ns_setup(ds4_engine*engine,const ds4_engine_options*opt,int context,int slots,
                         const char*disk_cache,ds4_ns_config**out,char*err,size_t cap){
    *out=NULL;const char*enabled=ds4_nf_env("DS4_NATIVE_SESSION");if(!enabled||!strcmp(enabled,"0"))return 0;
    if(strcmp(enabled,"1")||context!=1000001||slots!=1||(disk_cache&&*disk_cache)||
       !ds4_engine_is_deepseek41(engine)||opt->backend!=DS4_BACKEND_METAL||!opt->dspark||!opt->mtp_path||ds4_engine_mtp_draft_tokens(engine)<=1||
       opt->directional_steering_file||opt->load_slice||opt->tp.role!=DS4_TP_NONE||opt->distributed.role!=DS4_DISTRIBUTED_NONE||
       (ds4_nf_env("DS4_NATIVE_FRONTIER")&&strcmp(ds4_nf_env("DS4_NATIVE_FRONTIER"),"0")))
        return ds4_nf_error(err,cap,"session requires original local Metal DSpark, one1M slot, and no disk-cache restore");
    const char*host=ds4_nf_env("DS4_NATIVE_SESSION_IPV4"),*token=ds4_nf_env("DS4_NATIVE_SESSION_PAIR_TOKEN_FILE");
    const char*tier=ds4_nf_env("DS4_NATIVE_SESSION_TIER"),*techniques=ds4_nf_env("DS4_NATIVE_SESSION_TECHNIQUES");
    const char*path=ds4_nf_env("DS4_NATIVE_SESSION_QUALIFICATION"),*pin=ds4_nf_env("DS4_NATIVE_SESSION_QUALIFICATION_SHA256");
    const char*renderer=ds4_nf_env("DS4_NATIVE_SESSION_RENDERER_SHA256");
    const char*expected_producer=ds4_nf_env("DS4_NATIVE_SESSION_PRODUCER_SHA256");
    if(!host||strlen(host)>15||!token||!tier||!path||!pin||!ds4_nf_hex(pin,64)||!renderer||!ds4_nf_hex(renderer,64)||
       !expected_producer||!ds4_nf_hex(expected_producer,64)||
       (strcmp(tier,"accurate")&&strcmp(tier,"fast")&&strcmp(tier,"ultra")&&strcmp(tier,"hyper")))
        return ds4_nf_error(err,cap,"missing pinned native session configuration");
    ds4_ns_config*c=calloc(1,sizeof(*c));if(!c)return ds4_nf_error(err,cap,"session configuration allocation failed");
    c->http.dirfd=-1;uint32_t port;
    if(!ds4_nf_pair_token(token,c->http.token)||!ds4_nf_env_uint("DS4_NATIVE_SESSION_PORT",&port,1,65535)||
       !ds4_nf_env_uint("DS4_NATIVE_SESSION_DEADLINE_SECONDS",&c->http.deadline_seconds,60,7200)||
       inet_pton(AF_INET,host,&c->http.broker.sin_addr)!=1||!ds4_engine_native_frontier_identity(engine,&c->http.model))goto bad;
    strcpy(c->http.host,host);strcpy(c->tier,tier);c->http.broker.sin_family=AF_INET;c->http.broker.sin_port=htons((uint16_t)port);
    if(techniques){size_t n=strlen(techniques);if(n>=sizeof(c->techniques)||!ds4_ns_json_depth(techniques,n))goto bad;
        for(size_t i=0;i<n;i++)if((unsigned char)techniques[i]<32||(unsigned char)techniques[i]>126)goto bad;
        const char*p=techniques;json_ws(&p);if(*p!='{'||!json_skip_value(&p))goto bad;json_ws(&p);if(*p)goto bad;strcpy(c->techniques,techniques);}
    int fd=open(path,O_RDONLY|O_NOFOLLOW|O_CLOEXEC);struct stat st;ds4_nf_fields f;char sha[65];
    bool ok=fd>=0&&!fstat(fd,&st)&&S_ISREG(st.st_mode)&&st.st_uid==geteuid()&&st.st_nlink==1&&
        !(st.st_mode&0022)&&st.st_size>0&&st.st_size<=DS4_NF_QUALIFICATION_CAP&&!flock(fd,LOCK_SH|LOCK_NB);
    char*text=ok?malloc((size_t)st.st_size+1):NULL;if(ok&&!text)ok=false;
    if(ok){ok=pread(fd,text,(size_t)st.st_size,0)==st.st_size;text[st.st_size]=0;
        if(ok){ds4_nf_hash_bytes(text,(size_t)st.st_size,sha);ok=!strcmp(pin,sha)&&ds4_nf_parse_bounded(text,(size_t)st.st_size,&f,DS4_NF_QUALIFICATION_CAP);}}
    free(text);if(fd>=0)close(fd);
    const uint64_t required=(UINT64_C(1)<<0)|(UINT64_C(1)<<3)|(UINT64_C(1)<<6)|(UINT64_C(1)<<7)|
        (UINT64_C(1)<<11)|(UINT64_C(1)<<12)|(((UINT64_C(1)<<21)-1)&~((UINT64_C(1)<<14)-1));
    if(!ok||!ds4_nf_have(&f,required)||strcmp(f.schema,"dsv41-native-session-qualification-v2")||strcmp(f.model_sha256,DS4_NF_MODEL_SHA)||
       f.mode!=3||f.wire_version!=2||strcmp(f.tokenization,"native-final-token-ids-le32-v1")||
       strcmp(f.producer_binary_sha256,expected_producer)||
       !ds4_nf_hex(f.producer_source_revision,40)||!ds4_nf_hex(f.receiver_binary_sha256,64)||strcmp(f.renderer_sha256,renderer)||
       f.receiver_model_size!=c->http.model.size||f.receiver_model_device!=c->http.model.device||
       f.receiver_model_inode!=c->http.model.inode||f.receiver_model_mtime_ns!=c->http.model.mtime_ns)goto bad;
    char executable[PATH_MAX];uint32_t length=sizeof(executable);if(_NSGetExecutablePath(executable,&length))goto bad;
    fd=open(executable,O_RDONLY|O_CLOEXEC);ok=fd>=0&&ds4_nf_hash_file(fd,128ull<<20,sha,NULL)&&!strcmp(sha,f.receiver_binary_sha256);
    if(fd>=0)close(fd);if(!ok)goto bad;
    strcpy(c->http.producer_binary_sha256,f.producer_binary_sha256);strcpy(c->http.producer_source_revision,f.producer_source_revision);
    *out=c;ds4_ns_preserve_history_reasoning=true;fprintf(stderr,"native session enabled tier=%s context=1000001 chunk=49152 decode_delta_batch=32 custom_techniques=%s\n",tier,techniques?"explicit":"default");return 0;
bad:ds4_ns_config_free(c);return ds4_nf_error(err,cap,"native session configuration/model/build identity rejected");
}
static bool ds4_ns_conversation(const char*text,char out[65]){
    const char*p=text;bool found=false;json_ws(&p);if(*p++!='{')return false;
    for(;;){char*key=NULL;if(!json_string(&p,&key))return false;json_ws(&p);if(*p++!=':'){free(key);return false;}
        bool ours=!strcmp(key,"prompt_cache_key");free(key);
        if(ours){if(found||!ds4_nf_ascii(&p,out,65)||!*out)return false;found=true;
            for(const char*q=out;*q;q++)if(!isalnum((unsigned char)*q)&&*q!='-'&&*q!='_'&&*q!=':')return false;
        }else if(!json_skip_value(&p))return false;
        json_ws(&p);if(*p=='}'){p++;break;}if(*p++!=',')return false;}
    json_ws(&p);return found&&!*p;
}
static bool ds4_ns_history_policy(const char*text,bool*out){
    const char*p=text;bool found=false;*out=true;json_ws(&p);if(*p++!='{')return false;
    json_ws(&p);if(*p=='}'){p++;json_ws(&p);return !*p;}
    for(;;){char*key=NULL;if(!json_string(&p,&key))return false;json_ws(&p);if(*p++!=':'){free(key);return false;}
        bool ours=!strcmp(key,"include_previous_reasoning");free(key);json_ws(&p);
        if(ours){if(found)return false;found=true;
            if(!strncmp(p,"true",4)){*out=true;p+=4;}else if(!strncmp(p,"false",5)){*out=false;p+=5;}else return false;
        }else if(!json_skip_value(&p))return false;
        json_ws(&p);if(*p=='}'){p++;break;}if(*p++!=',')return false;
    }
    json_ws(&p);return !*p;
}
