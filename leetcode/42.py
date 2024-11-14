# 接雨水
from typing import List


def trap(height: List[int]) -> int:
    # 如果 height 为空，直接返回 0
    if not height:
        return 0

    # 获取 height 数组的长度
    n = len(height)
    # 初始化 left_max 和 right_max 数组，长度为 n，初始值为 0
    left_max = [0] * n
    right_max = [0] * n

    # 填充 left_max 数组
    # left_max[i] 表示从左到右扫描时，位置 i 及其左边的最大高度
    left_max[0] = height[0]
    for i in range(1, n):
        left_max[i] = max(left_max[i - 1], height[i])
    print(left_max)
    # 填充 right_max 数组
    # right_max[i] 表示从右到左扫描时，位置 i 及其右边的最大高度
    right_max[-1] = height[-1]
    for i in range(n - 2, -1, -1):
        right_max[i] = max(right_max[i + 1], height[i])
        # print(i, right_max[i], right_max[i + 1], height[i])    
    print(right_max)
    
    # 计算总的积水量
    total_water = 0
    for i in range(n):
        # 当前位置能存储的水量为 min(left_max[i], right_max[i]) - height[i]
        total_water += min(left_max[i], right_max[i]) - height[i]

    # 返回总的积水量
    return total_water

# 测试例子
height = [0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1]
expected_output = 6

# 调用函数并打印结果
output = trap(height)
print(f"{height}")
print(f"输出: {output}")
print(f"期望输出: {expected_output}")
print(f"测试通过: {output == expected_output}")