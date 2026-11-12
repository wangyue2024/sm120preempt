// Minimal BAR0 access for the RTX 5060 via sysfs resource0 (root only).
#pragma once
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>

#define BAR0_PATH "/sys/bus/pci/devices/0000:01:00.0/resource0"
#define BAR0_SIZE (64u << 20)

static volatile uint32_t *g_bar0;

static inline int bar0_open(void) {
    int fd = open(BAR0_PATH, O_RDWR | O_SYNC);
    if (fd < 0) { perror("open resource0"); return -1; }
    void *p = mmap(NULL, BAR0_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd);
    if (p == MAP_FAILED) { perror("mmap resource0"); return -1; }
    g_bar0 = (volatile uint32_t *)p;
    return 0;
}
static inline uint32_t rd32(uint32_t off) { return g_bar0[off >> 2]; }
static inline void wr32(uint32_t off, uint32_t v) { g_bar0[off >> 2] = v; }
