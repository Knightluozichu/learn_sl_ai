from typing import List

class Solution:
    def hIndex(self, citations: List[int]) -> int:
        hasH = len(citations)
        newList = sorted(citations)
        for i  in range(len(newList)):
            if newList[i] >= hasH:
                return hasH
            else:
                hasH -= 1
        return 0


def hIndex(citations):
    citations.sort()
    print(citations)
    n = len(citations)
    for h in range(n, -1, -1):
        count = 0
        for citation in citations:
            if citation >= h:
                count += 1
        if count >= h:
            return h
    return 0

citations = [3,0,6,1,5]
def test():
    solution = Solution()
    assert solution.hIndex(citations) == 3
    assert hIndex(citations) == 3

if "__main__" == __name__:
    test()
    print("274.py is ok")
                