# 14. 最长公共前缀
# 简单
# 相关标签
# 相关企业
# 编写一个函数来查找字符串数组中的最长公共前缀。

# 如果不存在公共前缀，返回空字符串 ""。

#  示例 1：

# 输入：strs = ["flower","flow","flight"]
# 输出："fl"
# 示例 2：

# 输入：strs = ["dog","racecar","car"]
# 输出：""
# 解释：输入不存在公共前缀。

from typing import List

# class Solution:
#     def longestCommonPrefix(self, strs: List[str]) -> str:
#         l = len(strs)
#         if l == 0:
#             return ""
#         if l == 1:
#             return strs[0]
#         prefix = strs[0]
#         for i in range(1,l):
#             prefix = self.commonPrefix(prefix,strs[i])
#             if not prefix:
#                 break
#         return prefix
    
#     def commonPrefix(self,str1,str2):
#         l1 = len(str1)
#         l2 = len(str2)
#         l = min(l1,l2)
#         i = 0
#         while i < l:
#             if str1[i] != str2[i]:
#                 break
#             i += 1
#         return str1[:i]
    

class Solution1:
    def longestCommonPrefix(self, strs: List[str]) -> str:
        if not strs:  # 如果数组为空，返回空字符串
            return ""
        
        # 初始化前缀为第一个字符串
        prefix = strs[0]
        
        # 从第二个字符串开始遍历
        for s in strs[1:]:
            # 更新前缀，直到找到当前字符串与前缀的最长公共前缀
            while not s.startswith(prefix):
                prefix = prefix[:-1]  # 每次缩短前缀，去掉最后一个字符
                if not prefix:
                    return ""  # 如果前缀变为空，说明没有公共前缀
        
        return prefix