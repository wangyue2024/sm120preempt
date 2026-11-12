// Minimal BAR0 access for NVIDIA Blackwell GPUs via sysfs resource0 (root only).
#pragma once
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <dirent.h>
#include <errno.h>

#define BAR0_DEFAULT_PATH "/sys/bus/pci/devices/0000:01:00.0/resource0"
#define BAR0_SIZE (64u << 20)

static volatile uint32_t *g_bar0 = NULL;

static inline const char *bar0_find_device_path(char *buf, size_t buf_sz) {
    // 1. Check environment variable override
    const char *env_path = getenv("BAR0_PATH");
    if (!env_path) env_path = getenv("XSCHED_SM120_BAR0_RESOURCE");
    if (env_path && access(env_path, R_OK | W_OK) == 0) {
        return env_path;
    }

    // 2. Check default common path
    if (access(BAR0_DEFAULT_PATH, R_OK | W_OK) == 0) {
        return BAR0_DEFAULT_PATH;
    }

    // 3. Auto-detect NVIDIA GPU in /sys/bus/pci/devices
    char first_candidate[512] = {0};
    DIR *dir = opendir("/sys/bus/pci/devices");
    if (dir) {
        struct dirent *ent;
        while ((ent = readdir(dir)) != NULL) {
            if (ent->d_name[0] == '.') continue;
            char path[512];
            snprintf(path, sizeof(path), "/sys/bus/pci/devices/%s/vendor", ent->d_name);
            FILE *f = fopen(path, "r");
            if (f) {
                char vstr[32] = {0};
                if (fgets(vstr, sizeof(vstr), f)) {
                    // 0x10de is NVIDIA Vendor ID
                    if (strstr(vstr, "0x10de") || strstr(vstr, "10de")) {
                        fclose(f);
                        // Check class
                        snprintf(path, sizeof(path), "/sys/bus/pci/devices/%s/class", ent->d_name);
                        FILE *fc = fopen(path, "r");
                        if (fc) {
                            char cstr[32] = {0};
                            if (fgets(cstr, sizeof(cstr), fc)) {
                                if (strstr(cstr, "0x03")) { // 0x0300 or 0x0302 (Display / 3D)
                                    fclose(fc);
                                    snprintf(buf, buf_sz, "/sys/bus/pci/devices/%s/resource0", ent->d_name);
                                    if (access(buf, R_OK | W_OK) == 0) {
                                        closedir(dir);
                                        return buf;
                                    }
                                    if (first_candidate[0] == '\0' && access(buf, F_OK) == 0) {
                                        snprintf(first_candidate, sizeof(first_candidate), "%s", buf);
                                    }
                                }
                            }
                            fclose(fc);
                        }
                        continue;
                    }
                }
                fclose(f);
            }
        }
        closedir(dir);
    }

    if (first_candidate[0] != '\0') {
        snprintf(buf, buf_sz, "%s", first_candidate);
        return buf;
    }

    return BAR0_DEFAULT_PATH;
}

static inline int bar0_open(void) {
    if (g_bar0 != NULL) return 0;

    char auto_buf[512];
    const char *target_path = bar0_find_device_path(auto_buf, sizeof(auto_buf));

    int fd = open(target_path, O_RDWR | O_SYNC);
    if (fd < 0) {
        fprintf(stderr, "[bar0] Failed to open BAR0 resource at %s: ", target_path);
        perror("");
        if (geteuid() != 0) {
            fprintf(stderr, "[bar0] HINT: BAR0 physical MMIO access requires ROOT privileges. Please re-run with 'sudo'!\n");
        } else if (errno == EROFS) {
            fprintf(stderr, "[bar0] HINT: /sys is mounted read-only (EROFS). If running in Docker, ensure container is started with '--privileged'!\n");
        }
        return -1;
    }
    void *p = mmap(NULL, BAR0_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd);
    if (p == MAP_FAILED) {
        fprintf(stderr, "[bar0] Failed to mmap BAR0 at %s: ", target_path);
        perror("");
        return -1;
    }
    g_bar0 = (volatile uint32_t *)p;
    return 0;
}

static inline uint32_t rd32(uint32_t off) { return g_bar0[off >> 2]; }
static inline void wr32(uint32_t off, uint32_t v) { g_bar0[off >> 2] = v; }
