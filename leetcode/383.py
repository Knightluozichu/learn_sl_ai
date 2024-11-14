# 赎金信
class Solution:
    def canConstruct(self, ransomNote:str, magazine:str) -> bool:
        # 思路
        # 1.遍历ransomNote，如果ransomNote中的字符在magazine中出现过，则将magazine中的字符删除
        # 2.如果ransomNote中的字符在magazine中没有出现过，则返回False
        # 3.遍历完ransomNote后返回True
        for char in ransomNote:
            if char in magazine:
                magazine = magazine.replace(char, '', 1)
            else:
                return False
        return True