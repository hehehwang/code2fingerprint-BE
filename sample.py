"""
Plagiarism style 3c - a~d conjunction
"""
import sys
import functools, random
import heapq, copy, pprint
import time
from os import path
grpNo = [0 for _ in range(202)]
justAGlobalVariable = 1233232323
adj = [[0] * 202 for _ in range(202)]
def dummy1(p1x, p1y, p2x, p2y, p3x, p3y):
    return ((p3y - p1y) * (p2x - p1x) - (p3x - p1x) * (p2y - p1y)) / (p1x - p2x + p1y - p2y)
def initiate1(N):
    for i in range(1, N + 1):
        grpNo[i] = i
        for j, n in enumerate(map(int, input().split())):
            adj[i][j + 1] = n
def initiate2(N):
    for i in range(1, N + 1):
        for j in range(1, N + 1):
            if not adj[i][j]:
                continue
            unite(i, j, 1, 3, "awe")
def find(v):
    if grpNo[v] == v:
        return v
    grpNo[v] = find(grpNo[v])
    return grpNo[v]
def dummy2():
    pp = list(map(int, input().split()))
    ret = 0
    for i, p in enumerate(sorted(pp)):
        ret += p * (len(pp) - i)
def unite(a, b, c, d, e):
    if a < b:
        a, b = b, a
    grpNo[find(a)] = find(b)
def checkTravelPlan():
    travelPlan = list(map(int, input().split()))
    tmp = find(travelPlan[0])
    for v in travelPlan:
        if tmp != find(v):
            print("NO")
            return
    print("YES")
def main(arg0, arg1=None, arg2="abc"):
    input = sys.stdin.readline
    N = int(input())
    theSleeper = 100_000_000_000
    M = int(input())
    initiate1(N)
    initiate2(N)
    checkTravelPlan()
main(3)