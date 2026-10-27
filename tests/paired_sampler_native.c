/* CPU-only check of the actual Mac request parser. Never opens a model. */
#define DS4_SERVER_TEST 1
#define DS4_SERVER_TEST_NO_MAIN 1
/* Only tokenization is stubbed: exercising the native JSON parser must not
 * require a second GGUF mapping alongside the owner's active engine. */
#define ds4_tokenize_rendered_chat paired_test_tokenize
#include "../engines/ds4-mac/ds4_server.c"
#undef ds4_tokenize_rendered_chat
void paired_test_tokenize(ds4_engine *e, const char *text, ds4_tokens *out) {
    (void)e;(void)text;memset(out,0,sizeof(*out));
}

static bool requested_sampling(const request *r) {
    return r->temperature_set && fabsf(r->temperature-.83f)<1e-6f &&
           r->top_p_set && fabsf(r->top_p-.87f)<1e-6f &&
           r->top_k_set && r->top_k==37 &&
           r->min_p_set && fabsf(r->min_p-.06f)<1e-6f &&
           r->seed==123456 && r->max_tokens==79;
}

int main(void) {
    request r;
    char err[256]={0};
    const char *body="{\"messages\":[{\"role\":\"user\",\"content\":\"q\"}],"
        "\"temperature\":0.83,\"top_p\":0.87,\"top_k\":37,\"min_p\":0.06,"
        "\"seed\":123456,\"max_tokens\":79,\"thinking\":true}";
    if (!parse_chat_request(NULL,NULL,body,128,1000001,&r,err,sizeof(err))) {
        fprintf(stderr,"Mac parser refused non-default samplers: %s\n",err);return 1;
    }
    if (!requested_sampling(&r)) {request_free(&r);return 2;}
    /* The same assertion must fail if a future proxy/parser drops to greedy. */
    r.temperature=0;r.top_p=1;r.top_k=0;r.min_p=0;
    if (requested_sampling(&r)) {request_free(&r);return 3;}
    request_free(&r);
    puts("PASS Mac native core samplers and greedy negative control; no model opened");
    return 0;
}
