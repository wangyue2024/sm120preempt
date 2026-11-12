// usage: bar0tool r <off> [count] | w <off> <val>
#include "bar0.h"
#include <string.h>
int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s r <off> [n] | w <off> <val>\n", argv[0]); return 2; }
    if (bar0_open()) return 1;
    uint32_t off = strtoul(argv[2], 0, 0);
    if (!strcmp(argv[1], "r")) {
        int n = argc > 3 ? atoi(argv[3]) : 1;
        for (int i = 0; i < n; i++) printf("%08x: %08x\n", off + 4 * i, rd32(off + 4 * i));
    } else if (!strcmp(argv[1], "w") && argc > 3) {
        uint32_t v = strtoul(argv[3], 0, 0);
        printf("%08x: %08x -> write %08x", off, rd32(off), v);
        wr32(off, v);
        printf(" -> %08x\n", rd32(off));
    }
    return 0;
}
