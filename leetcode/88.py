# %%
print("hello")
# %%
def merge(nums1, m, nums2, n):
    # 初始化指针
    p1, p2, p = m - 1, n - 1, m + n - 1
    
    # 从后向前遍历
    while p1 >= 0 and p2 >= 0:
        if nums1[p1] > nums2[p2]:
            nums1[p] = nums1[p1]
            p1 -= 1
        else:
            nums1[p] = nums2[p2]
            p2 -= 1
        p -= 1
    
    # 处理剩余元素
    # 如果 nums2 中还有剩余元素未处理完
    while p2 >= 0:
        nums1[p] = nums2[p2]
        p2 -= 1
        p -= 1

# 示例 1
nums1 = [1, 2, 3, 0, 0, 0]
m = 3
nums2 = [2, 5, 6]
n = 3
merge(nums1, m, nums2, n)
print(nums1)  # 输出: [1, 2, 2, 3, 5, 6]

# 示例 2
nums1 = [1]
m = 1
nums2 = []
n = 0
merge(nums1, m, nums2, n)
print(nums1)  # 输出: [1]