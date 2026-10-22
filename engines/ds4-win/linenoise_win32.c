#include "linenoise.h"

#if defined(_WIN32)

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <io.h>
#include <errno.h>
#include <limits.h>

/* Shared upstream UTF-8/grapheme logic and batched terminal writes. The
 * Windows editor keeps its existing blocking fallback; status refresh
 * reports unavailable so callers take their ordinary full refresh path. */
static struct {
    char *text;
    size_t len, cap;
    int fd, depth;
} terminal_update;

static int terminalWrite(int fd, const char *text, size_t len) {
    while (len) {
        int n = _write(fd, text, len > INT_MAX ? INT_MAX : (unsigned int)len);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) { if (!n) errno = EIO; return -1; }
        text += n;
        len -= (size_t)n;
    }
    return 0;
}

int linenoiseWrite(int fd, const char *text, size_t len) {
    if (!terminal_update.depth || terminal_update.fd != fd)
        return terminalWrite(fd, text, len);
    if (len > SIZE_MAX - terminal_update.len) { errno = ENOMEM; return -1; }
    size_t need = terminal_update.len + len;
    if (need > terminal_update.cap) {
        size_t cap = need <= SIZE_MAX / 2 ? need * 2 : need;
        char *p = realloc(terminal_update.text, cap);
        if (!p) return -1;
        terminal_update.text = p;
        terminal_update.cap = cap;
    }
    if (len) memcpy(terminal_update.text + terminal_update.len, text, len);
    terminal_update.len = need;
    return 0;
}

void linenoiseBeginUpdate(int fd) {
    if (terminal_update.depth++) return;
    terminal_update.fd = fd;
    terminal_update.len = 0;
    linenoiseWrite(fd, "\x1b[?2026h", 8);
}

void linenoiseEndUpdate(void) {
    if (!terminal_update.depth || terminal_update.depth > 1) {
        if (terminal_update.depth) terminal_update.depth--;
        return;
    }
    linenoiseWrite(terminal_update.fd, "\x1b[?2026l", 8);
    terminal_update.depth = 0;
    terminalWrite(terminal_update.fd, terminal_update.text, terminal_update.len);
    free(terminal_update.text);
    terminal_update.text = NULL;
    terminal_update.len = terminal_update.cap = 0;
}

static int utf8ByteLen(char c) {
    unsigned char uc = (unsigned char)c;
    if ((uc & 0x80) == 0)    return 1;   /* 0xxxxxxx: ASCII */
    if ((uc & 0xE0) == 0xC0) return 2;   /* 110xxxxx: 2-byte seq */
    if ((uc & 0xF0) == 0xE0) return 3;   /* 1110xxxx: 3-byte seq */
    if ((uc & 0xF8) == 0xF0) return 4;   /* 11110xxx: 4-byte seq */
    return 1; /* Fallback for invalid encoding, treat as single byte. */
}

/* Decode a UTF-8 sequence starting at 's' into a Unicode codepoint.
 * Returns the codepoint value. Assumes valid UTF-8 encoding. */
static uint32_t utf8DecodeChar(const char *s, size_t *len) {
    unsigned char *p = (unsigned char *)s;
    uint32_t cp;

    if ((*p & 0x80) == 0) {
        *len = 1;
        return *p;
    } else if ((*p & 0xE0) == 0xC0) {
        *len = 2;
        cp = (*p & 0x1F) << 6;
        cp |= (p[1] & 0x3F);
        return cp;
    } else if ((*p & 0xF0) == 0xE0) {
        *len = 3;
        cp = (*p & 0x0F) << 12;
        cp |= (p[1] & 0x3F) << 6;
        cp |= (p[2] & 0x3F);
        return cp;
    } else if ((*p & 0xF8) == 0xF0) {
        *len = 4;
        cp = (*p & 0x07) << 18;
        cp |= (p[1] & 0x3F) << 12;
        cp |= (p[2] & 0x3F) << 6;
        cp |= (p[3] & 0x3F);
        return cp;
    }
    *len = 1;
    return *p; /* Fallback for invalid sequences. */
}

/* Zero means more bytes are needed. Invalid sequences consume one byte. */
size_t linenoiseUtf8Decode(const char *s, size_t len, uint32_t *cp) {
    if (!len) return 0;
    unsigned char lead = (unsigned char)s[0];
    size_t n = (size_t)utf8ByteLen(s[0]);
    *cp = 0xfffd;
    if (lead < 0x80) { *cp = lead; return 1; }
    if (lead < 0xc2 || lead > 0xf4) return 1;
    for (size_t i = 1; i < n && i < len; i++)
        if (((unsigned char)s[i] & 0xc0) != 0x80) return 1;
    if (len < n) return 0;
    size_t decoded;
    uint32_t value = utf8DecodeChar(s, &decoded);
    if (value > 0x10ffff || (value >= 0xd800 && value <= 0xdfff) ||
        (n == 3 && value < 0x800) || (n == 4 && value < 0x10000)) return 1;
    *cp = value;
    return n;
}

/* Check if codepoint is a variation selector (emoji style modifiers). */
static int isVariationSelector(uint32_t cp) {
    return cp == 0xFE0E || cp == 0xFE0F;  /* Text/emoji style */
}

/* Check if codepoint is a skin tone modifier. */
static int isSkinToneModifier(uint32_t cp) {
    return cp >= 0x1F3FB && cp <= 0x1F3FF;
}

/* Check if codepoint is Zero Width Joiner. */
static int isZWJ(uint32_t cp) {
    return cp == 0x200D;
}

/* Check if codepoint is a Regional Indicator (for flag emoji). */
static int isRegionalIndicator(uint32_t cp) {
    return cp >= 0x1F1E6 && cp <= 0x1F1FF;
}

/* Check if codepoint is a combining mark or other zero-width character. */
static int isCombiningMark(uint32_t cp) {
    return (cp >= 0x0300 && cp <= 0x036F) ||   /* Combining Diacriticals */
           (cp >= 0x1AB0 && cp <= 0x1AFF) ||   /* Combining Diacriticals Extended */
           (cp >= 0x1DC0 && cp <= 0x1DFF) ||   /* Combining Diacriticals Supplement */
           (cp >= 0x20D0 && cp <= 0x20FF) ||   /* Combining Diacriticals for Symbols */
           (cp >= 0xFE20 && cp <= 0xFE2F);     /* Combining Half Marks */
}

/* Check if codepoint extends the previous character (doesn't start a new grapheme). */
static int isGraphemeExtend(uint32_t cp) {
    return isVariationSelector(cp) || isSkinToneModifier(cp) ||
           isZWJ(cp) || isCombiningMark(cp);
}

static int utf8CharWidth(uint32_t cp) {
    /* Control characters and combining marks: zero width. */
    if (cp < 32 || (cp >= 0x7F && cp < 0xA0)) return 0;
    if (isCombiningMark(cp)) return 0;

    /* Grapheme-extending characters: zero width.
     * These modify the preceding character rather than taking space. */
    if (isVariationSelector(cp)) return 0;
    if (isSkinToneModifier(cp)) return 0;
    if (isZWJ(cp)) return 0;

    /* Wide character ranges - these display as 2 columns:
     * - CJK Unified Ideographs and Extensions
     * - Fullwidth forms
     * - Various emoji ranges */
    if (cp >= 0x1100 &&
        (cp <= 0x115F ||                      /* Hangul Jamo */
         cp == 0x2329 || cp == 0x232A ||      /* Angle brackets */
         (cp >= 0x231A && cp <= 0x231B) ||    /* Watch, Hourglass */
         (cp >= 0x23E9 && cp <= 0x23F3) ||    /* Various symbols */
         (cp >= 0x23F8 && cp <= 0x23FA) ||    /* Various symbols */
         (cp >= 0x25AA && cp <= 0x25AB) ||    /* Small squares */
         (cp >= 0x25FB && cp <= 0x25FE) ||    /* Squares */
         (cp >= 0x2600 && cp <= 0x26FF) ||    /* Misc Symbols (sun, cloud, etc) */
         (cp >= 0x2700 && cp <= 0x27BF) ||    /* Dingbats (❤, ✂, etc) */
         (cp >= 0x2934 && cp <= 0x2935) ||    /* Arrows */
         (cp >= 0x2B05 && cp <= 0x2B07) ||    /* Arrows */
         (cp >= 0x2B1B && cp <= 0x2B1C) ||    /* Squares */
         cp == 0x2B50 || cp == 0x2B55 ||      /* Star, circle */
         (cp >= 0x2E80 && cp <= 0xA4CF &&
          cp != 0x303F) ||                    /* CJK ... Yi */
         (cp >= 0xAC00 && cp <= 0xD7A3) ||    /* Hangul Syllables */
         (cp >= 0xF900 && cp <= 0xFAFF) ||    /* CJK Compatibility Ideographs */
         (cp >= 0xFE10 && cp <= 0xFE1F) ||    /* Vertical forms */
         (cp >= 0xFE30 && cp <= 0xFE6F) ||    /* CJK Compatibility Forms */
         (cp >= 0xFF00 && cp <= 0xFF60) ||    /* Fullwidth Forms */
         (cp >= 0xFFE0 && cp <= 0xFFE6) ||    /* Fullwidth Signs */
         (cp >= 0x1F1E6 && cp <= 0x1F1FF) ||  /* Regional Indicators (flags) */
         (cp >= 0x1F300 && cp <= 0x1F64F) ||  /* Misc Symbols and Emoticons */
         (cp >= 0x1F680 && cp <= 0x1F6FF) ||  /* Transport and Map Symbols */
         (cp >= 0x1F900 && cp <= 0x1F9FF) ||  /* Supplemental Symbols */
         (cp >= 0x1FA00 && cp <= 0x1FAFF) ||  /* Chess, Extended-A */
         (cp >= 0x20000 && cp <= 0x2FFFF)))   /* CJK Extension B and beyond */
        return 2;

    return 1; /* Default: single width */
}

int linenoiseCharacterWidth(uint32_t cp) {
    return utf8CharWidth(cp);
}

size_t linenoiseNextGrapheme(const char *s, size_t len, int *width) {
    uint32_t cp;
    size_t used = linenoiseUtf8Decode(s, len, &cp);
    if (!used) { if (width) *width = 0; return 0; }
    int w = utf8CharWidth(cp), regional = isRegionalIndicator(cp), joined = 0;
    while (used < len) {
        size_t n = linenoiseUtf8Decode(s + used, len - used, &cp);
        if (!n) break;
        if (joined) {
            if (utf8CharWidth(cp) > w) w = utf8CharWidth(cp);
            joined = 0;
        } else if (isZWJ(cp)) {
            joined = 1;
        } else if (isGraphemeExtend(cp)) {
            if (cp == 0xfe0f && w) w = 2;
        } else if (regional && isRegionalIndicator(cp)) {
            regional = 0;
        } else break;
        used += n;
    }
    if (width) *width = w;
    return used;
}

int linenoiseRefreshStatus(struct linenoiseState *l) { (void)l; return 0; }

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
