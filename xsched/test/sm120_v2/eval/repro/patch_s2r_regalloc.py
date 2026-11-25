#!/usr/bin/env python3
"""Binary-patch every `S2R Rx, SR_TID.X` in an sm_120 cubin into `S2R Rx, SR_REGALLOC` (SR id 0x3d).
Used to measure what the hardware reports in SR_REGALLOC (DEF-20). Usage: patch_s2r_regalloc.py in.cubin out.cubin"""
import struct, sys
b = bytearray(open(sys.argv[1], 'rb').read()); n = 0
for off in range(0, len(b) - 16, 16):
    w0, w1 = struct.unpack_from('<QQ', b, off)
    if (w0 & 0xffff) == 0x7919 and ((w1 >> 8) & 0xff) == 0x21:
        struct.pack_into('<QQ', b, off, w0, (w1 & ~(0xff << 8)) | (0x3d << 8)); n += 1
open(sys.argv[2], 'wb').write(b); print("patched S2R instructions:", n)
