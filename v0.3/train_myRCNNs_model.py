# IOU Part 1
def if_intersection(xmin_a, xmax_a, ymin_a, ymax_a, xmin_b, xmax_b, ymin_b, ymax_b):
    if_intersect = False
    # 通过四条if来查看两个方框是否有交集。如果四种状况都不存在，我们视为无交集
    if xmin_a < xmax_b <= xmax_a and (ymin_a < ymax_b <= ymax_a or ymin_a <= ymin_b < ymax_a):
        if_intersect = True
    elif xmin_a <= xmin_b < xmax_a and (ymin_a < ymax_b <= ymax_a or ymin_a <= ymin_b < ymax_a):
        if_intersect = True
    elif xmin_b < xmax_a <= xmax_b and (ymin_b < ymax_a <= ymax_b or ymin_b <= ymin_a < ymax_b):
        if_intersect = True
    elif xmin_b <= xmin_a < xmax_b and (ymin_b < ymax_a <= ymax_b or ymin_b <= ymin_a < ymax_b):
        if_intersect = True
    else:
        return False
    # 在有交集的情况下，我们通过大小关系整理两个方框各自的四个顶点， 通过它们得到交集面积
    if if_intersect == True:
        x_sorted_list = sorted([xmin_a, xmax_a, xmin_b, xmax_b])
        y_sorted_list = sorted([ymin_a, ymax_a, ymin_b, ymax_b])
        x_intersect_w = x_sorted_list[2] - x_sorted_list[1]
        y_intersect_h = y_sorted_list[2] - y_sorted_list[1]
        area_inter = x_intersect_w * y_intersect_h
        return area_inter
 
# IOU Part 2
def IOU(ver1, vertice2):
    # vertices in four points
    # 整理输入顶点
    vertice1 = [ver1[0], ver1[1], ver1[0]+ver1[2], ver1[1]+ver1[3]]
    area_inter = if_intersection(vertice1[0], vertice1[2], vertice1[1], vertice1[3], vertice2[0], vertice2[2], vertice2[1], vertice2[3])
    # 如果有交集，计算IOU
    if area_inter:
        area_1 = ver1[2] * ver1[3]
        area_2 = vertice2[4] * vertice2[5]
        iou = float(area_inter) / (area_1 + area_2 - area_inter)
        return iou
    return False


# Read in data and save data for Alexnet
def load_train_proposals(datafile, num_clss, threshold = 0.5, svm = False, save=False, save_path='dataset.pkl'):
    train_list = open(datafile,'r')
    labels = []
    images = []
    for line in train_list:
        tmp = line.strip().split(' ')
        # tmp0 = image address
        # tmp1 = label
        # tmp2 = rectangle vertices
        img = skimage.io.imread(tmp[0])
        # python的selective search函数
        img_lbl, regions = selectivesearch.selective_search(img, scale=500, sigma=0.9, min_size=10)
        candidates = set()
        for r in regions:
        # excluding same rectangle (with different segments)
            # 剔除重复的方框
            if r['rect'] in candidates:
                continue
            # 剔除太小的方框
        if r['size'] < 220:
            continue
        # resize to 224 * 224 for input
            # 重整方框的大小
            proposal_img, proposal_vertice = clip_pic(img, r['rect'])
        # Delete Empty array
            # 如果截取后的图片为空，剔除
        if len(proposal_img) == 0:
            continue
            # Ignore things contain 0 or not C contiguous array
        x, y, w, h = r['rect']
        # 长或宽为0的方框，剔除
        if w == 0 or h == 0:
            continue
            # Check if any 0-dimension exist
            # image array的dim里有0的，剔除
        [a, b, c] = np.shape(proposal_img)
        if a == 0 or b == 0 or c == 0:
            continue
        im = Image.fromarray(proposal_img)
        resized_proposal_img = resize_image(im, 224, 224)
        candidates.add(r['rect'])
        img_float = pil_to_nparray(resized_proposal_img)
        images.append(img_float)
            # 计算IOU
        ref_rect = tmp[2].split(',')
        ref_rect_int = [int(i) for i in ref_rect]
        iou_val = IOU(ref_rect_int, proposal_vertice)
            # labels, let 0 represent default class, which is background
        index = int(tmp[1])
        if svm == False:
            label = np.zeros(num_clss+1)
            if iou_val < threshold:
                label[0] = 1
            else:
                label[index] = 1
            labels.append(label)
        else:
            if iou_val < threshold:
                labels.append(0)
            else:
                labels.append(index)
    if save:
        pickle.dump((images, labels), open(save_path, 'wb'))
    return images, labels








