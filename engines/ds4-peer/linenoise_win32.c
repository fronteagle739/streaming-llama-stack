#include "linenoise.h"

#if defined(_WIN32)

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static char linenoise_more_sentinel;
char *linenoiseEditMore = &linenoise_more_sentinel;

int linenoiseEditStart(struct linenoiseState *l, int stdin_fd, int stdout_fd,
                       char *buf, size_t buflen, const char *prompt) {
    (void)l;
    (void)stdin_fd;
    (void)stdout_fd;
    (void)buf;
    (void)buflen;
    (void)prompt;
    return -1;
}

char *linenoiseEditFeed(struct linenoiseState *l) {
    (void)l;
    return NULL;
}

char *linenoiseEditFeedByte(struct linenoiseState *l, char c) {
    (void)l;
    (void)c;
    return NULL;
}

int linenoiseEditQueueInput(struct linenoiseState *l, const char *buf, size_t len) {
    (void)l;
    (void)buf;
    (void)len;
    return 0;
}

int linenoiseEditInsert(struct linenoiseState *l, const char *c, size_t clen) {
    (void)l;
    (void)c;
    (void)clen;
    return 0;
}

size_t linenoiseEditQueuedInput(struct linenoiseState *l) {
    (void)l;
    return 0;
}

void linenoiseEditClear(struct linenoiseState *l) { (void)l; }
int linenoiseEditSetStatus(struct linenoiseState *l, const char *status,
                           const char *start_escape, const char *end_escape) {
    (void)l;
    (void)status;
    (void)start_escape;
    (void)end_escape;
    return 0;
}
void linenoiseEditSetLayoutCallback(struct linenoiseState *l,
                                    linenoiseLayoutCallback *fn,
                                    void *privdata) {
    (void)l;
    (void)fn;
    (void)privdata;
}
void linenoiseEditStop(struct linenoiseState *l) { (void)l; }
void linenoiseHide(struct linenoiseState *l) { (void)l; }
void linenoiseShow(struct linenoiseState *l) { (void)l; }

char *linenoise(const char *prompt) {
    if (prompt) {
        fputs(prompt, stdout);
        fflush(stdout);
    }

    size_t cap = 256;
    size_t len = 0;
    char *line = (char *)malloc(cap);
    if (!line) return NULL;

    for (;;) {
        int c = fgetc(stdin);
        if (c == EOF) {
            free(line);
            return NULL;
        }
        if (c == '\n') break;
        if (c == '\r') {
            int next = fgetc(stdin);
            if (next != '\n' && next != EOF) ungetc(next, stdin);
            break;
        }
        if (len + 1 >= cap) {
            size_t next_cap = cap * 2;
            char *next = (char *)realloc(line, next_cap);
            if (!next) {
                free(line);
                return NULL;
            }
            line = next;
            cap = next_cap;
        }
        line[len++] = (char)c;
    }
    line[len] = '\0';
    return line;
}

void linenoiseFree(void *ptr) { free(ptr); }
void linenoiseSetCompletionCallback(linenoiseCompletionCallback *fn) { (void)fn; }
void linenoiseSetHintsCallback(linenoiseHintsCallback *fn) { (void)fn; }
void linenoiseSetFreeHintsCallback(linenoiseFreeHintsCallback *fn) { (void)fn; }
void linenoiseAddCompletion(linenoiseCompletions *lc, const char *str) {
    (void)lc;
    (void)str;
}
int linenoiseHistoryAdd(const char *line) {
    (void)line;
    return 0;
}
int linenoiseHistorySetMaxLen(int len) {
    (void)len;
    return 0;
}
int linenoiseHistorySave(const char *filename) {
    (void)filename;
    return 0;
}
int linenoiseHistoryLoad(const char *filename) {
    (void)filename;
    return 0;
}
void linenoiseClearScreen(void) { fputs("\033[H\033[2J", stdout); }
void linenoiseSetMultiLine(int ml) { (void)ml; }
void linenoisePrintKeyCodes(void) {}
void linenoiseMaskModeEnable(void) {}
void linenoiseMaskModeDisable(void) {}
void linenoiseRestoreRawMode(void) {}

#endif /* _WIN32 */
