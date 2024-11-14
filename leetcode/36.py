# 36. 有效的数独
# 中等
# 相关标签
# 相关企业
# 请你判断一个 9 x 9 的数独是否有效。只需要 根据以下规则 ，验证已经填入的数字是否有效即可。

# 数字 1-9 在每一行只能出现一次。
# 数字 1-9 在每一列只能出现一次。
# 数字 1-9 在每一个以粗实线分隔的 3x3 宫内只能出现一次。（请参考示例图）
 

# 注意：

# 一个有效的数独（部分已被填充）不一定是可解的。
# 只需要根据以上规则，验证已经填入的数字是否有效即可。
# 空白格用 '.' 表示。

class Solution:
    def isValidSudoku(self, board):
        # 思路
        # 1.横向遍历每一行记录是否除了'.'以外有重复的数字
        # 2.纵向遍历每一行记录是否除了'.'以外有重复的数字
        # 3.每3个为一轮，每3横遍历纵向3个数为一轮，记录是否除了'.'以外有重复的数字，
        # 4.但凡1~3出现重复的数字就返回False，else返回True
        rows = [set() for _ in range(9)]
        cols = [set() for _ in range(9)]
        boxes = [set() for _ in range(9)]

        for i in range(9):
            for j in range(9):
                num = board[i][j]
                if num!= '.':
                    box_index = (i // 3) * 3 + j // 3
                    if num in rows[i] or num in cols[j] or num in boxes[box_index]:
                        return False
                    rows[i].add(num)
                    cols[j].add(num)
                    boxes[box_index].add(num)
        return True
