#!/usr/bin/env python3
# tolerant SM120 disassembler: batch nvdisasm, replace undecodable 16B words with NOP and retry
import sys, subprocess, struct, tempfile, os, re
NVD="/home/wy/cuda-12.9/bin/nvdisasm"
NOP=struct.pack("<QQ",0x7918,0x000fc00000000000)
def run(buf,tmp):
    open(tmp,"wb").write(buf)
    p=subprocess.run([NVD,"-b","SM120",tmp],capture_output=True,text=True)
    return p.returncode==0 and "error" not in p.stderr, p.stdout
def ok(buf,tmp): return run(buf,tmp)[0]
def main():
    data=bytearray(open(sys.argv[1],"rb").read()); data=data[:len(data)//16*16]
    base=int(sys.argv[2],0) if len(sys.argv)>2 else 0
    tmp=tempfile.mktemp(suffix=".bin"); bad=set()
    n=len(data)//16
    while True:
        good,out=run(bytes(data),tmp)
        if good: break
        lo,hi=0,n   # find smallest prefix that fails
        while hi-lo>1:
            mid=(lo+hi)//2
            if ok(bytes(data[:mid*16]),tmp): lo=mid
            else: hi=mid
        idx=hi-1; bad.add(idx); data[idx*16:idx*16+16]=NOP
    orig=open(sys.argv[1],"rb").read()
    for l in out.splitlines():
        m=re.match(r"\s*/\*([0-9a-f]{4,})\*/\s*(.*)",l)
        if not m: continue
        off=int(m.group(1),16); s=m.group(2).split("&req")[0].split("?")[0].strip()
        i=off//16; w0,w1=struct.unpack("<QQ",orig[off:off+16])
        if i in bad: s="??? undecodable"
        elif s=="NOP;": continue
        print("%05x  %016x %016x  %s"%(base+off,w0,w1,s))
    os.unlink(tmp)
main()
