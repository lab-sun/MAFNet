# By Zhen FENG, Aug. 5, 2022
# Email: zfeng@outlook.com

import numpy as np 
from PIL import Image 
 
def get_palette():
    unlabelled = [177,0,178]
    pothole = [231,230,0]

    palette    = np.array([unlabelled, pothole])
    return palette

def visualize(image_name, predictions, weight_name):
    palette = get_palette()
    for (i, pred) in enumerate(predictions):
        pred = predictions[i].cpu().numpy()
        img = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
        for cid in range(0, len(palette)): # fix the mistake from the MFNet code on Dec.27, 2019
            img[pred == cid] = palette[cid]
        img = Image.fromarray(np.uint8(img))
        img.save('runs/Pred_' + weight_name + '_' + image_name[i] + '.png')

def getScores(conf_matrix):
    if conf_matrix.sum() == 0:
        return 0, 0, 0, 0, 0
    with np.errstate(divide='ignore',invalid='ignore'):
        globalacc = np.diag(conf_matrix).sum() / np.float(conf_matrix.sum())
        classpre = np.diag(conf_matrix) / conf_matrix.sum(0).astype(np.float)
        classrecall = np.diag(conf_matrix) / conf_matrix.sum(1).astype(np.float)
        IU = np.diag(conf_matrix) / (conf_matrix.sum(1) + conf_matrix.sum(0) - np.diag(conf_matrix)).astype(np.float)
        pre_p = classpre[1]
        recall_p = classrecall[1]
        iou_p = IU[1]
        F_score_p = 2*(recall_p*pre_p)/(recall_p+pre_p)

        pre_b = classpre[0]
        recall_b = classrecall[0]
        iou_b = IU[0]
        F_score_b = 2*(recall_b*pre_b)/(recall_b+pre_b)
    return globalacc, pre_p, recall_p, F_score_p, iou_p, pre_b, recall_b, F_score_b, iou_b