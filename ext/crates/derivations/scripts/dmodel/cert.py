import sys
sys.argv = ['dproto.py', sys.argv[1], '0']
src = open('dproto.py').read().split("print(f\"{nv} variables")[0]
exec(src)
piv = {}
for idx, (m, c, lab) in enumerate(EQ):
    prov = 1 << idx
    while m:
        p = (m & -m).bit_length() - 1
        if p in piv:
            pm, pc, pp = piv[p]
            m ^= pm; c ^= pc; prov ^= pp
        else:
            piv[p] = (m, c, prov)
            break
    else:
        if c:
            print("contradiction from", bin(prov).count('1'), "equations:")
            for k in bits(prov):
                mm, cc, ll = EQ[k]
                print("  ", ll, "const", cc, [NAME[v] for v in bits(mm)])
            break
