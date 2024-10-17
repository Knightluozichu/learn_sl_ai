# %% 1768. Merge Strings Alternately
class Solution:
    def mergeAlternately(self, word1: str, word2: str) -> str:
        index=0
        str_sum=""
        for w1 in word1:
            str_sum += w1
            if index < len(word2):
                w2 = word2[index]
                index+=1
                str_sum += w2
        if index <= len(word2):
            str_sum += word2[index:]
        return str_sum

word1 = "abc"
word2 = "pqr"

solution = Solution()
print(solution.mergeAlternately(word1, word2))

word1 = "ab"
word2 = "pqrs"

# solution = Solution()
print(solution.mergeAlternately(word1, word2))

word1 = "abcd"
word2 = "pq"

# solution = Solution()
print(solution.mergeAlternately(word1, word2))
# %%
class Solution:
    def mergeAlternately(self, word1:str, word2:str)->str:
        str_sum = ""
        len1, len2=len(word1), len(word2)
        index=0
        while index<len1 and index<len2:
            str_sum += word1[index]+word2[index]
            index+=1
        str_sum += word1[index:] + word2[index:]
        return str_sum

word1 = "abc"
word2 = "pqr"
s = Solution()
print(s.mergeAlternately(word1, word2))

word1 = "ab"
word2 = "pqrs"
print(s.mergeAlternately(word1, word2))

word1 = "abcd"
word2 = "pq"
print(s.mergeAlternately(word1, word2))
# %% 1071. Greatest Common Divisor of Strings
class Solution:
    def gcdOfStrings(self, str1: str, str2: str) -> str:
        if str1+str2 != str2+str1:
            return ""
        from math import gcd
        return str1[:gcd(len(str1), len(str2))]
    
str1 = "ABCABC"
str2 = "ABC"
solution = Solution()
print(solution.gcdOfStrings(str1, str2))

# %%
