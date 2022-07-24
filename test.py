from sys import stdin 
from select import select

i, o, e = select([stdin], [], [], 1)

if (i):
    print("you said", stdin.readline().strip())
else:
    print("you said nothing")