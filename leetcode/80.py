# 80. 删除有序数组中的重复项 II
# 中等
# 相关标签
# 相关企业
# 给你一个有序数组 nums ，请你 原地 删除重复出现的元素，使得出现次数超过两次的元素只出现两次 ，返回删除后数组的新长度。

# 不要使用额外的数组空间，你必须在 原地 修改输入数组 并在使用 O(1) 额外空间的条件下完成
from typing import List


class Solution:
    def removeDuplicates(self, nums: List[int]) -> int:
        l = len(nums)
        if l <=2:
            return 2
        j = 2
        for i in range(2,l):
            if nums[i] != nums[j-2]:
                nums[j] = nums[i]
                j += 1
        return j



