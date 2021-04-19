import json
from PIL import Image, ImageDraw
import os
import numpy as np
import matplotlib.pyplot as plt

preds_path = "../data/hw02_preds"
data_path = '../data/RedLights2011_Medium'
gts_path = '../data/hw02_annotations'
score_thresh = 0.2

def main():
    with open(os.path.join(preds_path,'preds_test.json'),'r') as f:
        preds_test = json.load(f)

    with open(os.path.join(preds_path,'heatmaps_test.json'),'r') as f:
        heatmaps_test = json.load(f)

    with open(os.path.join(gts_path, 'annotations_test.json'),'r') as f:
        gts_test = json.load(f)

    for filename, pred_boxes in preds_test.items():
        gt_boxes = gts_test[filename]
        heatmap = np.array(heatmaps_test[filename])

        # Get image
        I = Image.open(os.path.join(data_path,filename))
        d = ImageDraw.Draw(I)

        # Visualize heatmap
        plt.imshow(heatmap, cmap='cool', interpolation='nearest')
        plt.savefig(os.path.join("../data/hw02_preds/", filename.split(".")[0]) + "_heatmap.jpg")

        # Draw prediction boxes
        for box in pred_boxes:
            tl_row, tl_col, br_row, br_col, score = box
            pred_color = (0,0,255)

            if score > score_thresh:
                d.rectangle(((tl_col, tl_row), (br_col, br_row)), outline=pred_color)
                d.text((br_col+1, br_row+1), str(round(score, 2)))

        # Draw gt_boxes
        for box in gt_boxes:
            tl_row, tl_col, br_row, br_col = box
            gt_color = (0,255,0)

            d.rectangle(((tl_col, tl_row), (br_col, br_row)), outline=gt_color)

        # Save image with boxes
        I.save(os.path.join("../data/hw02_preds/", filename.split(".")[0]) + "_result.jpg")


if __name__ == "__main__":
    main()
