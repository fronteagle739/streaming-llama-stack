#ifndef DS4_NATIVE_SESSION_RECOVERY_H
#define DS4_NATIVE_SESSION_RECOVERY_H
/* Recovery text and policy for the persistent RTX prefill (pure; CPU-tested).
   - The RTX's own refusal text (HTTP error body "detail", or the session
     status "error") is carried to the Mac log, the chat error and the notice.
   - A failed RTX prefill falls back to Mac-local prefill for that turn, with a
     visible notice, unless the turn was cancelled or is a qualification run
     that must exercise the RTX.
   - A conversation whose native state was invalidated is reset and rebuilt by
     re-prefilling its history, with a visible notice. */
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>
#define DS4_NS_DETAIL_CAP 512u
#define DS4_NS_NOTICE_CAP 384u
/* Printable ASCII that is safe inside a JSON string and a log line: '"' and
   '\\' become '\'', control and non-ASCII bytes become ' ', and a cut text
   ends in "...". out is always terminated. */
static void ds4_ns_safe_text(char*out,size_t cap,const char*in,size_t n){
    if(!out||!cap)return;out[0]=0;if(!in)return;
    size_t o=0;bool cut=false;
    for(size_t i=0;i<n&&in[i];i++){
        if(o+1>=cap){cut=true;break;}
        unsigned char c=(unsigned char)in[i];
        out[o++]=c=='"'||c=='\\'?'\'':c<32||c>=127?' ':(char)c;
    }
    out[o]=0;
    if(cut&&cap>4){size_t k=o<cap-4?o:cap-4;memcpy(out+k,"...",4);}
}
static int ds4_ns_hex(char c){return c>='0'&&c<='9'?c-'0':c>='a'&&c<='f'?c-'a'+10:c>='A'&&c<='F'?c-'A'+10:-1;}
/* Decode the JSON string value of key (first occurrence, any depth) into
   safe text. False when absent or malformed. */
static bool ds4_ns_json_field(const char*body,size_t n,const char*key,char*out,size_t cap){
    char pat[48];int k=snprintf(pat,sizeof(pat),"\"%s\"",key);if(k<=0||(size_t)k>=sizeof(pat))return false;
    for(size_t i=0;i+(size_t)k<=n;i++){
        if(memcmp(body+i,pat,(size_t)k))continue;
        size_t j=i+(size_t)k;while(j<n&&(body[j]==' '||body[j]=='\t'||body[j]=='\r'||body[j]=='\n'))j++;
        if(j>=n||body[j]!=':')continue;j++;
        while(j<n&&(body[j]==' '||body[j]=='\t'||body[j]=='\r'||body[j]=='\n'))j++;
        if(j>=n||body[j]!='"')continue;j++;
        char raw[2048];size_t o=0;
        while(j<n&&body[j]!='"'){
            char c=body[j++];
            if(c=='\\'){
                if(j>=n)return false;char e=body[j++];
                if(e=='n'||e=='t'||e=='r'||e=='b'||e=='f')c=' ';
                else if(e=='u'){
                    if(j+4>n)return false;int v=0;
                    for(int d=0;d<4;d++){int h=ds4_ns_hex(body[j+d]);if(h<0)return false;v=v*16+h;}
                    j+=4;c=v>=32&&v<127?(char)v:'?';
                }else c=e;
            }
            if(o+1<sizeof(raw))raw[o++]=c;
        }
        if(j>=n)return false;
        ds4_ns_safe_text(out,cap,raw,o);return true;
    }
    return false;
}
/* The refusal text of an HTTP error reply: FastAPI's {"detail":...}, else a
   status "error", else the raw body; prefixed with the status. */
static void ds4_ns_http_detail(const char*body,size_t n,int status,char*out,size_t cap){
    char text[DS4_NS_DETAIL_CAP];
    if(!body||!n)snprintf(text,sizeof(text),"(empty body)");
    else if(!ds4_ns_json_field(body,n,"detail",text,sizeof(text))&&!ds4_ns_json_field(body,n,"error",text,sizeof(text)))
        ds4_ns_safe_text(text,sizeof(text),body,n);
    snprintf(out,cap,"RTX HTTP %d: %s",status,text);
}
static bool ds4_ns_should_fallback(bool cancelled,bool force_remote){return !cancelled&&!force_remote;}
static void ds4_ns_fallback_notice(char*out,size_t cap,const char*detail){
    char d[256];ds4_ns_safe_text(d,sizeof(d),detail&&detail[0]?detail:"no reason given",DS4_NS_DETAIL_CAP);
    snprintf(out,cap,"RTX busy (%s), prefilling on Mac: slower",d);
}
static void ds4_ns_rebuild_notice(char*out,size_t cap,const char*detail){
    char d[200];ds4_ns_safe_text(d,sizeof(d),detail&&detail[0]?detail:"earlier native failure",DS4_NS_DETAIL_CAP);
    snprintf(out,cap,"Rebuilding conversation state (%s): re-prefilling history",d);
}
/* Append a second notice to a first one ("; " separated, bounded). */
static void ds4_ns_notice_append(char*notice,size_t cap,const char*more){
    if(!notice||!cap||!more||!more[0])return;size_t n=strlen(notice);
    if(!n){snprintf(notice,cap,"%s",more);return;}
    if(n+2<cap)snprintf(notice+n,cap-n,"; %s",more);
}
#endif
