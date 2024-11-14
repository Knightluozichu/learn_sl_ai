# 1. 盛最多水的容器
# 中等
# 相关标签
# 相关企业
# 提示
# 给定一个长度为 n 的整数数组 height 。有 n 条垂线，第 i 条线的两个端点是 (i, 0) 和 (i, height[i]) 。

# 找出其中的两条线，使得它们与 x 轴共同构成的容器可以容纳最多的水。

# 返回容器可以储存的最大水量。

# 说明：你不能倾斜容器。

from typing import List


# class Solution:
#     def maxArea(self, height: List[int]) -> int:
#         area = 0
#         left = 0
#         right = len(height) - 1
#         for i in range(len(height)):
#             area = max(area, min(height[left], height[right]) * (right - left))
#             if height[left] < height[right]:
#                 left += 1
#             else:
#                 right -= 1
#         return area

class Solution:
    def maxArea(self, height: List[int]) -> int:
        left, right = 0, len(height) - 1
        max_area = 0
        best_left, best_right = 0, len(height) - 1  # 用于记录盛水最多的两条线的位置

        while left < right:
            # 计算当前面积
            current_area = (right - left) * min(height[left], height[right])

            # 更新最大面积并记录最优的两个指针位置
            if current_area > max_area:
                max_area = current_area
                best_left, best_right = left, right

            # 移动较短的线的指针
            if height[left] < height[right]:
                left += 1
            else:
                right -= 1

        # 返回最大面积以及对应的两个线的位置
        return max_area, (best_left, best_right)