// xsched-bar0d: narrow privileged proxy for SM120 Level-3 trap pulses (DEF-10).
//
// Started as root. Validates the GPU, maps ONLY the two BAR0 pages that hold the trap
// register and the flush register, creates a group-restricted SOCK_SEQPACKET socket, then
// drops every privilege (setgroups/setgid/setuid to --run-as, no_new_privs, not dumpable).
// After that it serves one operation, PULSE, with a per-connection token-bucket rate limit.
// Pulses from all clients are serialized by the single-threaded event loop, so pulse
// sequences of different processes can never interleave.
//
// usage: xsched-bar0d [--bdf dddd:bb:ss.f] [--group xsched] [--socket PATH] [--run-as nobody]
//                     [--rate 5000] [--burst 64] [--verbose]

#include <algorithm>
#include <cerrno>
#include <csignal>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <dirent.h>
#include <fcntl.h>
#include <grp.h>
#include <pwd.h>
#include <string>
#include <sys/epoll.h>
#include <sys/mman.h>
#include <sys/prctl.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <unistd.h>
#include <unordered_map>

#include "xsched/cuda/hal/level3/sm120_bar0_proxy.h"

namespace
{

using namespace xsched::cuda::sm120;
namespace bp = xsched::cuda::sm120::bar0proxy;

constexpr size_t kPage = 4096;
constexpr int kMaxClients = 1024;
volatile sig_atomic_t g_stop = 0;
bool g_verbose = false;

void Log(const char *fmt, ...)
{
    char ts[32];
    time_t now = time(nullptr);
    struct tm tmv;
    localtime_r(&now, &tmv);
    strftime(ts, sizeof(ts), "%F %T", &tmv);
    fprintf(stderr, "[xsched-bar0d %s] ", ts);
    va_list ap;
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
    fputc('\n', stderr);
    fflush(stderr);
}

[[noreturn]] void Die(const char *fmt, ...)
{
    fprintf(stderr, "[xsched-bar0d] fatal: ");
    va_list ap;
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
    fputc('\n', stderr);
    exit(1);
}

bool ReadSysfs(const std::string &path, char *buf, size_t len)
{
    FILE *f = fopen(path.c_str(), "r");
    if (!f) return false;
    const bool ok = fgets(buf, static_cast<int>(len), f) != nullptr;
    fclose(f);
    return ok;
}

bool IsNvidiaGpu(const std::string &bdf)
{
    char v[32] = {}, c[32] = {};
    const std::string base = "/sys/bus/pci/devices/" + bdf;
    if (!ReadSysfs(base + "/vendor", v, sizeof(v)) || !ReadSysfs(base + "/class", c, sizeof(c))) return false;
    return strncmp(v, "0x10de", 6) == 0 && strncmp(c, "0x03", 4) == 0;
}

std::string DiscoverSingleGpu()
{
    std::string found;
    int n = 0;
    if (DIR *d = opendir("/sys/bus/pci/devices")) {
        while (struct dirent *e = readdir(d)) {
            if (e->d_name[0] == '.') continue;
            if (IsNvidiaGpu(e->d_name)) { found = e->d_name; ++n; }
        }
        closedir(d);
    }
    if (n != 1) Die("found %d NVIDIA GPUs; pass --bdf (one daemon per GPU)", n);
    return found;
}

double NowSec()
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return static_cast<double>(ts.tv_sec) + static_cast<double>(ts.tv_nsec) * 1e-9;
}

struct Client
{
    pid_t pid = 0;
    uid_t uid = 0;
    double tokens = 0.0;
    double last = 0.0;
    double last_limit_log = 0.0;
    uint64_t pulses = 0;
    uint64_t limited = 0;
};

void OnSignal(int) { g_stop = 1; }

} // namespace

int main(int argc, char **argv)
{
    std::string bdf, group = "xsched", sock_path, run_as = "nobody";
    double rate = 5000.0, burst = 64.0;
    for (int i = 1; i < argc; ++i) {
        auto arg = [&](const char *name) -> const char * {
            if (strcmp(argv[i], name) != 0) return nullptr;
            if (i + 1 >= argc) Die("%s needs a value", name);
            return argv[++i];
        };
        if (const char *v = arg("--bdf")) bdf = v;
        else if (const char *v = arg("--group")) group = v;
        else if (const char *v = arg("--socket")) sock_path = v;
        else if (const char *v = arg("--run-as")) run_as = v;
        else if (const char *v = arg("--rate")) rate = atof(v);
        else if (const char *v = arg("--burst")) burst = atof(v);
        else if (strcmp(argv[i], "--verbose") == 0) g_verbose = true;
        else Die("unknown argument %s", argv[i]);
    }
    if (rate <= 0.0 || burst < 1.0) Die("--rate must be > 0 and --burst >= 1");
    if (geteuid() != 0) Die("must be started as root (it maps BAR0, then drops privileges)");

    if (bdf.empty()) bdf = DiscoverSingleGpu();
    if (bdf.size() >= sizeof(bp::Reply::bdf) || !IsNvidiaGpu(bdf)) Die("%s is not an NVIDIA GPU", bdf.c_str());
    if (sock_path.empty()) sock_path = std::string(bp::kDefaultDir) + "/bar0-" + bdf + ".sock";

    struct group *gr = getgrnam(group.c_str());
    if (!gr) Die("group '%s' does not exist (create it and add the users allowed to preempt)", group.c_str());
    const gid_t sock_gid = gr->gr_gid;
    struct passwd *pw = getpwnam(run_as.c_str());
    if (!pw || pw->pw_uid == 0) Die("--run-as '%s' must be an existing non-root user", run_as.c_str());
    const uid_t drop_uid = pw->pw_uid;
    const gid_t drop_gid = pw->pw_gid;

    // Map only the two pages the pulse touches.
    const std::string res = "/sys/bus/pci/devices/" + bdf + "/resource0";
    int fd = open(res.c_str(), O_RDWR | O_SYNC | O_CLOEXEC);
    if (fd < 0) Die("open %s: %s", res.c_str(), strerror(errno));
    auto map_page = [&](uint32_t off) {
        void *p = mmap(nullptr, kPage, PROT_READ | PROT_WRITE, MAP_SHARED, fd, off & ~(kPage - 1));
        if (p == MAP_FAILED) Die("mmap BAR0+0x%x: %s", off, strerror(errno));
        return reinterpret_cast<volatile uint32_t *>(static_cast<uint8_t *>(p) + (off & (kPage - 1)));
    };
    volatile uint32_t *trap_reg = map_page(kBar0TrapRegOff);
    volatile uint32_t *ptimer = map_page(kBar0PtimerLoOff);
    close(fd);

    // Socket: directory root:root 0755, socket root:<group> 0660.
    const std::string dir = sock_path.substr(0, sock_path.rfind('/'));
    if (!dir.empty() && mkdir(dir.c_str(), 0755) != 0 && errno != EEXIST) Die("mkdir %s: %s", dir.c_str(), strerror(errno));
    struct stat st;
    if (lstat(sock_path.c_str(), &st) == 0) {
        if (!S_ISSOCK(st.st_mode)) Die("%s exists and is not a socket", sock_path.c_str());
        unlink(sock_path.c_str());
    }
    int lfd = socket(AF_UNIX, SOCK_SEQPACKET | SOCK_CLOEXEC | SOCK_NONBLOCK, 0);
    if (lfd < 0) Die("socket: %s", strerror(errno));
    struct sockaddr_un addr = {};
    addr.sun_family = AF_UNIX;
    if (sock_path.size() >= sizeof(addr.sun_path)) Die("socket path too long");
    memcpy(addr.sun_path, sock_path.c_str(), sock_path.size() + 1);
    const mode_t old_umask = umask(0177);
    if (bind(lfd, reinterpret_cast<struct sockaddr *>(&addr), sizeof(addr)) != 0) Die("bind %s: %s", sock_path.c_str(), strerror(errno));
    umask(old_umask);
    if (chown(sock_path.c_str(), 0, sock_gid) != 0 || chmod(sock_path.c_str(), 0660) != 0) Die("chown/chmod socket: %s", strerror(errno));
    if (listen(lfd, 128) != 0) Die("listen: %s", strerror(errno));

    // Drop every privilege. The BAR0 mappings and the listening socket stay valid.
    if (setgroups(0, nullptr) != 0 || setgid(drop_gid) != 0 || setuid(drop_uid) != 0) Die("drop privileges: %s", strerror(errno));
    if (setuid(0) == 0 || geteuid() == 0) Die("privileges were not dropped");
    prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0);
    prctl(PR_SET_DUMPABLE, 0, 0, 0, 0);

    signal(SIGPIPE, SIG_IGN);
    struct sigaction sa = {};
    sa.sa_handler = OnSignal;
    sigaction(SIGTERM, &sa, nullptr);
    sigaction(SIGINT, &sa, nullptr);

    int ep = epoll_create1(EPOLL_CLOEXEC);
    struct epoll_event ev = {};
    ev.events = EPOLLIN;
    ev.data.fd = lfd;
    epoll_ctl(ep, EPOLL_CTL_ADD, lfd, &ev);
    Log("serving %s on %s (group %s, rate %.0f/s burst %.0f per connection, running as %s)",
        bdf.c_str(), sock_path.c_str(), group.c_str(), rate, burst, run_as.c_str());

    std::unordered_map<int, Client> clients;
    uint64_t total_pulses = 0;
    auto drop = [&](int cfd, const char *why) {
        const Client &c = clients[cfd];
        if (g_verbose || c.limited) Log("disconnect pid %d uid %u: %s (pulses %llu, rate-limited %llu)", c.pid, c.uid, why,
                                        (unsigned long long)c.pulses, (unsigned long long)c.limited);
        epoll_ctl(ep, EPOLL_CTL_DEL, cfd, nullptr);
        close(cfd);
        clients.erase(cfd);
    };

    struct epoll_event evs[64];
    while (!g_stop) {
        int n = epoll_wait(ep, evs, 64, 1000);
        if (n < 0) {
            if (errno == EINTR) continue;
            Die("epoll_wait: %s", strerror(errno));
        }
        for (int i = 0; i < n; ++i) {
            const int fdv = evs[i].data.fd;
            if (fdv == lfd) {
                for (;;) {
                    int cfd = accept4(lfd, nullptr, nullptr, SOCK_CLOEXEC | SOCK_NONBLOCK);
                    if (cfd < 0) break;
                    struct ucred cr = {};
                    socklen_t cl = sizeof(cr);
                    if (static_cast<int>(clients.size()) >= kMaxClients ||
                        getsockopt(cfd, SOL_SOCKET, SO_PEERCRED, &cr, &cl) != 0) {
                        close(cfd);
                        continue;
                    }
                    Client c;
                    c.pid = cr.pid;
                    c.uid = cr.uid;
                    c.tokens = burst;
                    c.last = NowSec();
                    clients[cfd] = c;
                    struct epoll_event cev = {};
                    cev.events = EPOLLIN | EPOLLRDHUP;
                    cev.data.fd = cfd;
                    epoll_ctl(ep, EPOLL_CTL_ADD, cfd, &cev);
                    if (g_verbose) Log("connect pid %d uid %u", cr.pid, cr.uid);
                }
                continue;
            }
            if (evs[i].events & (EPOLLHUP | EPOLLERR | EPOLLRDHUP)) {
                drop(fdv, "closed");
                continue;
            }
            bp::Request rq = {};
            const ssize_t got = recv(fdv, &rq, sizeof(rq), 0);
            if (got == 0) { drop(fdv, "closed"); continue; }
            if (got < 0) { if (errno != EAGAIN && errno != EINTR) drop(fdv, "recv error"); continue; }
            Client &c = clients[fdv];
            bp::Reply rp = {};
            rp.magic = bp::kMagic;
            rp.seq = rq.seq;
            snprintf(rp.bdf, sizeof(rp.bdf), "%s", bdf.c_str());
            if (got != static_cast<ssize_t>(sizeof(rq)) || rq.magic != bp::kMagic || rq.version != bp::kVersion) {
                rp.status = bp::kBadRequest;
            } else if (rq.op == bp::kOpHello) {
                rp.status = bp::kOk;
            } else if (rq.op == bp::kOpPulse) {
                const double now = NowSec();
                c.tokens = std::min(burst, c.tokens + (now - c.last) * rate);
                c.last = now;
                if (c.tokens >= 1.0) {
                    c.tokens -= 1.0;
                    FireBar0Pulse(trap_reg, ptimer);
                    ++c.pulses;
                    ++total_pulses;
                    rp.status = bp::kOk;
                } else {
                    ++c.limited;
                    rp.status = bp::kRateLimited;
                    if (now - c.last_limit_log > 1.0) {
                        Log("rate limit: pid %d uid %u (%llu pulses refused so far)", c.pid, c.uid, (unsigned long long)c.limited);
                        c.last_limit_log = now;
                    }
                }
            } else {
                rp.status = bp::kBadRequest;
            }
            rp.pulses = c.pulses;
            if (send(fdv, &rp, sizeof(rp), MSG_NOSIGNAL | MSG_DONTWAIT) != static_cast<ssize_t>(sizeof(rp))) {
                drop(fdv, "send failed");
            }
        }
    }
    Log("stopping (%llu pulses served)", (unsigned long long)total_pulses);
    return 0;
}
