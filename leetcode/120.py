from typing import List


class Solution:
    def minimumTotal(self, triangle: List[List[int]]) -> int:
        if not triangle:
            return 0
        
        # 初始化 dp 为三角形最后一行的值
        dp = triangle[-1]
        
        # 从倒数第二行开始向上遍历
        for i in range(len(triangle) - 2, -1, -1):
            for j in range(len(triangle[i])):
                # 更新 dp[j] 为当前点的最小路径和
                dp[j] = triangle[i][j] + min(dp[j], dp[j + 1])
        
        # dp[0] 中存储的是最小路径和
        return dp[0]
    

triangle = [[2],[3,4],[6,5,7],[4,1,8,3]]
s = Solution()
print(s.minimumTotal(triangle))  # 11


