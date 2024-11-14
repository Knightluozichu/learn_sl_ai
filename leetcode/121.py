from typing import List


class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        # 思路
        # 从左遍历 ，最小，
        # 从右遍历，最大，
        # 落差值
        maxLeft = prices.copy()
        maxRight = prices.copy()

        for i in range(1,len(prices),1):
            if prices[i] > maxLeft[i-1]:
                maxLeft[i] = maxLeft[i-1]
        for i in range(len(prices)-2,-1,-1):
            # print(i,prices[i],maxRight[i+1])
            if prices[i] < maxRight[i+1]:
                maxRight[i] = maxRight[i+1]
        # print(maxLeft)
        # print(maxRight)
        sum = 0
        for i in range(len(prices)):
            drop =  maxRight[i] - maxLeft[i]
            print(i,drop,sum)
            if drop > sum:
                sum = drop
        # print(sum)
        return sum

# test
# 示例 1：

# 输入：[7,1,5,3,6,4]
# 输出：5
# 解释：在第 2 天（股票价格 = 1）的时候买入，在第 5 天（股票价格 = 6）的时候卖出，最大利润 = 6-1 = 5 。
#      注意利润不能是 7-1 = 6, 因为卖出价格需要大于买入价格；同时，你不能在买入前卖出股票。

if __name__ == '__main__':
    # test code
    assert Solution().maxProfit([7,1,5,3,6,4]) == 5
    assert Solution().maxProfit([7,6,4,3,1]) == 0

# 示例 2：

# 输入：prices = [7,6,4,3,1]
# 输出：0
# 解释：在这种情况下, 没有交易完成, 所以最大利润为 0。