#!/usr/bin/env python3

for i in range(1, 64):
    cmd = "hipcc -fPIC --shared libfoo%d.cpp -o libfoo%d.so" % (i, i)
    print(cmd)
