from typing import List

class Solution:
    def containsNearbyDuplicate(self, nums: List[int], k: int) -> bool:
        num_dict = {}
        for i, num in enumerate(nums):
            if num in num_dict and i - num_dict[num] <= k:
                return True
            num_dict[num] = i
        return False

# class Solution:
#     def containsNearbyDuplicate(self, nums: List[int], k: int) -> bool:
#         # 当前需要判断的num，
#         # 遍历nums，判断除当前num位置后的k个
#         # 如果有满足return true，
#         # 直至所有数字遍历完，如果都不满足return false

#         for i in range(len(nums)):
#             for j in range(i + 1, i + k + 1):
#                 if j < len(nums) and nums[i] == nums[j]:
#                     return True
#         return False
    
# 示例 1：

# 输入：nums = [1,2,3,1], k = 3
# 输出：true
# 示例 2：

# 输入：nums = [1,0,1,1], k = 1
# 输出：true
# 示例 3：

# 输入：nums = [1,2,3,1,2,3], k = 2
# 输出：false

nums = [1,2,3,1]
k = 3
s = Solution()
print(s.containsNearbyDuplicate(nums, k))  # true

nums = [1,0,1,1]
k = 1
s = Solution()
print(s.containsNearbyDuplicate(nums, k))  # true

nums = [1,2,3,1,2,3]
k = 2
s = Solution()
print(s.containsNearbyDuplicate(nums, k))  # false