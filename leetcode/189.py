from typing import List

class Solution:
    def rotate(self, nums: List[int], k: int) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        l = len(nums)
        k = k % l  # 处理 k 大于数组长度的情况
        nums[:] = nums[-k:] + nums[:-k]  # 原地修改 nums

# 示例 1:
nums = [1,2,3,4,5,6,7]
k = 3
s = Solution()
s.rotate(nums, k)
print(nums)  # [5,6,7,1,2,3,4]

# 示例 2:
nums = [-1,-100,3,99]
k = 2
s.rotate(nums, k)
print(nums)  # [3,99,-1,-100]

#include <iostream>
#include <vector>
#include <algorithm>

# class Solution {
# public:
#     void rotate(std::vector<int>& nums, int k) {
#         int l = nums.size();
#         k = k % l;  // 处理 k 大于数组长度的情况
#         std::reverse(nums.begin(), nums.end());
#         std::reverse(nums.begin(), nums.begin() + k);
#         std::reverse(nums.begin() + k, nums.end());
#     }
# };

# int main() {
#     // 示例 1:
#     std::vector<int> nums1 = {1, 2, 3, 4, 5, 6, 7};
#     int k1 = 3;
#     Solution().rotate(nums1, k1);
#     for (int num : nums1) {
#         std::cout << num << " ";
#     }
#     std::cout << std::endl;  // 输出: 5 6 7 1 2 3 4

#     // 示例 2:
#     std::vector<int> nums2 = {-1, -100, 3, 99};
#     int k2 = 2;
#     Solution().rotate(nums2, k2);
#     for (int num : nums2) {
#         std::cout << num << " ";
#     }
#     std::cout << std::endl;  // 输出: 3 99 -1 -100

#     return 0;
# }