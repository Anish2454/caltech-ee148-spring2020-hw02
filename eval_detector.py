import os
import json
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def compute_iou(box_1, box_2):
    '''
    This function takes a pair of bounding boxes and returns intersection-over-
    union (IoU) of two bounding boxes.
    '''

    # Get coordinates of intersecting box
    tli_row = max(box_1[0], box_2[0])
    tli_col = max(box_1[1], box_2[1])
    bri_row = min(box_1[2], box_2[2])
    bri_col = min(box_1[3], box_2[3])

    # Calculate are of intersecting box
    heighti = max(bri_row - tli_row, 0)
    widthi = max(bri_col - tli_col, 0)
    intersection_area = heighti * widthi
    if intersection_area == 0:
        return 0

    # Get area of union
    box_1_height = box_1[2] - box_1[0]
    box_1_width = box_1[3] - box_1[1]
    box_2_height = box_2[2] - box_2[0]
    box_2_width = box_2[3] - box_1[1]
    box_1_area = box_1_height * box_1_width
    box_2_area = box_2_height * box_2_width

    iou = float(intersection_area) / (box_1_area + box_2_area - intersection_area)

    assert (iou >= 0) and (iou <= 1.0)

    return iou


def compute_counts(preds, gts, iou_thr=0.5, conf_thr=0.5):
    '''
    This function takes a pair of dictionaries (with our JSON format; see ex.)
    corresponding to predicted and ground truth bounding boxes for a collection
    of images and returns the number of true positives, false positives, and
    false negatives.
    <preds> is a dictionary containing predicted bounding boxes and confidence
    scores for a collection of images.
    <gts> is a dictionary containing ground truth bounding boxes for a
    collection of images.
    '''
    TP = 0
    FP = 0
    FN = 0

    '''
    BEGIN YOUR CODE
    '''
    for pred_file, pred in preds.items():
        gt = gts[pred_file]
        pred = [x for x in pred if float(x[4]) >= conf_thr]

        for i in range(len(gt)):
            if len(pred) == 0:
                FN += (len(gt) - i)
                break

            ious = np.zeros(len(pred))
            for j in range(len(pred)):
                iou = compute_iou(pred[j][:4], gt[i])
                ious[j] = iou

            # get position of max
            max_iou = max(ious)
            max_iou_ix = ious.tolist().index(max_iou)

            if max_iou > iou_thr:
                # match the gt to this pred as a TP
                TP += 1
                # remove this pred from preds so we don't double count it
                pred.pop(max_iou_ix)
            else:
                # if not, this gt is a FN
                FN += 1

        # at the end, number of extra preds is FP
        FP += len(pred)


    '''
    END YOUR CODE
    '''

    return TP, FP, FN

# set a path for predictions and annotations:
preds_path = '../data/hw02_preds'
gts_path = '../data/hw02_annotations'

# load splits:
split_path = '../data/hw02_splits'
file_names_train = np.load(os.path.join(split_path,'file_names_train.npy'))
file_names_test = np.load(os.path.join(split_path,'file_names_test.npy'))

# Set this parameter to True when you're done with algorithm development:
done_tweaking = True

'''
Load training data.
'''
with open(os.path.join(preds_path,'preds_train.json'),'r') as f:
    preds_train = json.load(f)

with open(os.path.join(gts_path, 'annotations_train.json'),'r') as f:
    gts_train = json.load(f)

if done_tweaking:

    '''
    Load test data.
    '''

    with open(os.path.join(preds_path,'preds_test.json'),'r') as f:
        preds_test = json.load(f)

    with open(os.path.join(gts_path, 'annotations_test.json'),'r') as f:
        gts_test = json.load(f)


# For a fixed IoU threshold, vary the confidence thresholds.
# The code below gives an example on the training set for one IoU threshold.
iou_threshs = [0.25, 0.5, 0.75]
confidence_thrs = []
for fname in preds_train:
    for box in preds_train[fname]:
        confidence_thrs.append(float(box[4]))
confidence_thrs = np.random.choice(confidence_thrs, 100)
confidence_thrs = np.sort(confidence_thrs)

for iou_thresh in tqdm(iou_threshs):
    tp_train = np.zeros(len(confidence_thrs))
    fp_train = np.zeros(len(confidence_thrs))
    fn_train = np.zeros(len(confidence_thrs))
    for i, conf_thr in tqdm(enumerate(confidence_thrs), leave=False):
        tp_train[i], fp_train[i], fn_train[i] = compute_counts(preds_train, gts_train, iou_thr=iou_thresh, conf_thr=conf_thr)

    # Plot training set PR curves
    P = tp_train / (tp_train + fp_train)
    R = tp_train / (tp_train + fn_train)

    plt.plot(R,P, '-o', markersize=2)

plt.legend(["IOU Thresh 0.25", "IOU Thresh 0.5", "IOU Thresh 0.75"])
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.savefig('train_PR_curve.png')

if done_tweaking:
    print('Code for plotting test set PR curves.')
    plt.figure()

    # For a fixed IoU threshold, vary the confidence thresholds.
    # The code below gives an example on the training set for one IoU threshold.
    iou_threshs = [0.25, 0.5, 0.75]
    confidence_thrs = []
    for fname in preds_test:
        for box in preds_test[fname]:
            confidence_thrs.append(float(box[4]))
    confidence_thrs = np.random.choice(confidence_thrs, 100)
    confidence_thrs = np.sort(confidence_thrs)

    for iou_thresh in tqdm(iou_threshs):
        tp_test = np.zeros(len(confidence_thrs))
        fp_test = np.zeros(len(confidence_thrs))
        fn_test = np.zeros(len(confidence_thrs))
        for i, conf_thr in tqdm(enumerate(confidence_thrs), leave=False):
            tp_test[i], fp_test[i], fn_test[i] = compute_counts(preds_test, gts_test, iou_thr=iou_thresh, conf_thr=conf_thr)

        # Plot training set PR curves
        P = tp_test / (tp_test + fp_test)
        R = tp_test / (tp_test + fn_test)

        plt.plot(R,P, '-o', markersize=2)

    plt.legend(["IOU Thresh 0.25", "IOU Thresh 0.5", "IOU Thresh 0.75"])
    plt.savefig('test_PR_curve.png')
