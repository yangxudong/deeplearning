from __future__ import print_function
import math


def score(l1, l2, p=0.98):
    """
        Calculates Ranked Biased Overlap (RBO) score.
        l1 -- Ranked List 1
        l2 -- Ranked List 2
    """
    if l1 == None: l1 = []
    if l2 == None: l2 = []

    sl, ll = sorted([(len(l1), l1), (len(l2), l2)])
    s, S = sl
    l, L = ll
    if s == 0: return 0

    # Calculate the overlaps at ranks 1 through l
    # (the longer of the two lists)
    ss = set([])  # contains elements from the smaller list till depth i
    ls = set([])  # contains elements from the longer list till depth i
    x_d = {0: 0}
    sum1 = 0.0
    for i in range(l):
        x = L[i]
        y = S[i] if i < s else None
        d = i + 1

        # if two elements are same then
        # we don't need to add to either of the set
        if x == y:
            x_d[d] = x_d[d - 1] + 1.0
        # else add items to respective list
        # and calculate overlap
        else:
            ls.add(x)
            if y != None: ss.add(y)
            x_d[d] = x_d[d - 1] + (1.0 if x in ss else 0.0) + (1.0 if y in ls else 0.0)
            # calculate average overlap
        sum1 += x_d[d] / d * pow(p, d)

    sum2 = 0.0
    for i in range(l - s):
        d = s + i + 1
        sum2 += x_d[d] * (d - s) / (d * s) * pow(p, d)

    sum3 = ((x_d[l] - x_d[s]) / l + x_d[s] / s) * pow(p, l)

    # Equation 32
    rbo_ext = (1 - p) / p * (sum1 + sum2) + sum3
    return rbo_ext


def rbo_score(l1, l2, p):
    if not l1 or not l2:
        return 0
    s1 = set()
    s2 = set()
    max_depth = len(l1)
    score = 0.0
    for d in range(max_depth):
        s1.add(l1[d])
        s2.add(l2[d])
        avg_overlap = len(s1 & s2) / (d + 1)
        score += math.pow(p, d) * avg_overlap
    return (1 - p) * score


if __name__ == "__main__":
    list1 = ['0', '1', '2', '3', '4', '5']
    list2 = ['1', '0', '2', '3', '4', '5']
    list3 = ['0', '1', '2', '3', '5', '4']
    list4 = ['2', '1', '0', '3', '5', '4']
    print(rbo_score(list1, list2, 0.01))
    print(rbo_score(list1, list3, 0.01))
    print(rbo_score(list1, list2, 0.5))
    print(rbo_score(list1, list3, 0.5))
    print(rbo_score(list1, list4, 0.5))
    print(rbo_score(list1, list2, 0.9))
    print(rbo_score(list1, list3, 0.9))
    print("-----------------------------------")
    print(rbo_score(list1, list1, 0))
    print(rbo_score(list1, list1, 0.5))
    print(rbo_score(list1, list1, 0.9))
    print("-----------------------------------")

    list1 = ['0', '1', '3', '4', '5']
    list2 = ['1', '0', '3', '4', '5']
    list3 = ['0', '1', '3', '5', '4']
    print(rbo_score(list1, list2, 0.5))
    print(rbo_score(list1, list3, 0.5))
    # print score(list1, list2, p = 0.90)

    # list1 = ['012']
    # list2 = []
    # print rbo_score(list1, list2, p = 0.98)
