# 1431. 拥有最多糖果的孩子
# 简单
# 相关标签
# 相关企业
# 提示
# 给你一个数组 candies 和一个整数 extraCandies ，其中 candies[i] 代表第 i 个孩子拥有的糖果数目。

# 对每一个孩子，检查是否存在一种方案，将额外的 extraCandies 个糖果分配给孩子们之后，此孩子有 最多 的糖果。注意，允许有多个孩子同时拥有 最多 的糖果数目。

'''
解题思路：
max_candies = max(candies)
new_candies = [candy + extraCandies >= max_candies for candy in candies]
return new_candies
'''

from typing import List


class Solution:
    def kidsWithCandies(self, candies:List[int], extraCandies:int)->List[bool]:
        max_candies = max(candies)
        return [candy + extraCandies >= max_candies for candy in candies]