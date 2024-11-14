# 加载/data/labelme/000306.jpg
# %%
import matplotlib.pyplot as plt
import matplotlib.patches as patches

True_bbox = [237.009,41.921,305.147,87.019]
predection_bbox = [230.009,31.921,295.147,83.]
# predection_bbox = True_bbox

img = plt.imread("data/labelme/000306.jpg")
fig = plt.imshow(img)

# Create a Rectangle patch
fig.axes.add_patch(plt.Rectangle((True_bbox[0], True_bbox[1]), True_bbox[2] - True_bbox[0], True_bbox[3] - True_bbox[1], linewidth=2, edgecolor='blue', facecolor='none'))

fig.axes.add_patch(plt.Rectangle((predection_bbox[0], predection_bbox[1]), predection_bbox[2] - predection_bbox[0], predection_bbox[3] - predection_bbox[1], linewidth=2, edgecolor='r', facecolor='none'))

# %%
import numpy as np

def Iou(box1,box2,wh=False):
    if wh==False:
        xmin1,ymin1,xmax1,ymax1 = box1
        xmin2,ymin2,xmax2,ymax2 = box2
    else:
        xmin1,ymin1,xmax1,ymax1 = box1[0],box1[1],box1[0]+box1[2],box1[1]+box1[3]
        xmin2,ymin2,xmax2,ymax2 = box2[0],box2[1],box2[0]+box2[2],box2[1]+box2[3]
    #计算交集部分左上角坐标
    xmin = max(xmin1,xmin2)
    ymin = max(ymin1,ymin2)
    #计算交集部分右下角坐标
    xmax = min(xmax1,xmax2)
    ymax = min(ymax1,ymax2)
    #计算交集面积
    inter_area = max(0,xmax-xmin)*max(0,ymax-ymin)
    #计算并集面积
    '''
    在计算并集面积时，减去交集区域是为了避免重复计算。让我详细解释一下这个过程。

    为什么要减去交集面积？

    当两个区域（如边界框或掩码）有重叠部分时，如果直接将它们的面积相加，重叠的部分会被计算两次。为了得到准确的并集面积，我们需要从总和中减去一次交集面积。
    '''
    union_area = (xmax1-xmin1)*(ymax1-ymin1)+(xmax2-xmin2)*(ymax2-ymin2)-inter_area
    #计算IOU
    iou = inter_area/union_area
    return iou

def NMS(bboxes,threshold=0.5):
    # 按照置信度从高到低排序边界框
    bboxes = sorted(bboxes, key=lambda x: x[4], reverse=True)
    # 存储经过NMS处理后的边界框
    bboxes_after_nms = []
    # 当还有未处理的边界框时
    while len(bboxes) > 0:
        # 选择置信度最高的边界框
        chosen_box = bboxes.pop(0)
        # 将选择的边界框加入结果列表
        bboxes_after_nms.append(chosen_box)
        # 过滤掉与选择的边界框IOU大于阈值的边界框
        bboxes = [box for box in bboxes if Iou(chosen_box, box) < threshold]
    # 返回经过NMS处理后的边界框
    return bboxes_after_nms

def draw_bbox(img,bboxes):
    fig = plt.imshow(img)
    for box in bboxes:
        xmin,ymin,xmax,ymax = box[:4]
        fig.axes.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=2, edgecolor='r', facecolor='none'))

def mAp(true_bboxes,prediction_bboxes,threshold=0.5):
    mAp = 0
    for i in range(len(true_bboxes)):
        true_bbox = true_bboxes[i]
        prediction_bbox = prediction_bboxes[i]
        iou = Iou(true_bbox,prediction_bbox)
        if iou>threshold:
            mAp += 1
    mAp /= len(true_bboxes)
    return mAp
# 计算True_bbox和predection_bbox IOU
print(Iou(True_bbox,predection_bbox))
print(mAp([True_bbox],[predection_bbox]))

# %%
