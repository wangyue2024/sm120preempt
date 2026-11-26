#!/usr/bin/env python3
"""DEF-10 protocol checks for xsched-bar0d that never touch hardware (HELLO and malformed requests only).

usage: test_bar0_proxy_p34.py <socket> [hello_iters]
Prints one PASS/FAIL line per check and the HELLO round-trip latency (the IPC cost a proxied pulse adds).
"""
import socket
import struct
import sys
import time

MAGIC, VERSION, HELLO, PULSE = 0x50304258, 1, 1, 2
REQ, REP = struct.Struct("<IHHQ"), struct.Struct("<IiQQ16s")


def conn(path):
    s = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    s.settimeout(1.0)
    s.connect(path)
    return s


def rt(s, raw):
    s.send(raw)
    return REP.unpack(s.recv(64))


def main():
    path = sys.argv[1]
    iters = int(sys.argv[2]) if len(sys.argv) > 2 else 20000
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"[BAR0-PROXY {name}] {'PASS' if cond else 'FAIL'} {detail}")

    s = conn(path)
    m, st, seq, pulses, bdf = rt(s, REQ.pack(MAGIC, VERSION, HELLO, 7))
    check("hello", m == MAGIC and st == 0 and seq == 7, f"bdf={bdf.rstrip(bytes(1)).decode()}")
    # Malformed requests: wrong magic, wrong version, unknown op, short packet. None may pulse.
    for name, raw in (("bad-magic", REQ.pack(0x1234, VERSION, PULSE, 1)), ("bad-version", REQ.pack(MAGIC, 9, PULSE, 2)),
                      ("bad-op", REQ.pack(MAGIC, VERSION, 77, 3)), ("short", b"XB")):
        m, st, seq, pulses, _ = rt(s, raw)
        check(name, st == -1 and pulses == 0, f"status={st} pulses={pulses}")
    # The connection survives malformed requests.
    m, st, seq, _, _ = rt(s, REQ.pack(MAGIC, VERSION, HELLO, 8))
    check("survives", st == 0 and seq == 8)
    # Many concurrent clients, then abrupt disconnects.
    cs = [conn(path) for _ in range(64)]
    good = sum(rt(c, REQ.pack(MAGIC, VERSION, HELLO, i))[1] == 0 for i, c in enumerate(cs))
    for c in cs:
        c.close()
    check("64-clients", good == 64, f"{good}/64")
    # HELLO round trip = the IPC cost a proxied pulse adds.
    raw = REQ.pack(MAGIC, VERSION, HELLO, 1)
    lat = []
    for _ in range(iters):
        t = time.perf_counter_ns()
        s.send(raw)
        s.recv(64)
        lat.append(time.perf_counter_ns() - t)
    lat.sort()
    p = lambda q: lat[int(q * (len(lat) - 1))] / 1000
    print(f"[BAR0-PROXY rtt] iters={iters} p50={p(0.5):.1f} us p99={p(0.99):.1f} us max={lat[-1] / 1000:.1f} us")
    print("[BAR0-PROXY all]", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
