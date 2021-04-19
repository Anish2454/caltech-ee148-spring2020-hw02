import os
import numpy as np
import json
from PIL import Image
from tqdm import tqdm

def compute_convolution(I, T, stride=None, weak=False):
    '''
    This function takes an image <I> and a template <T> (both numpy arrays)
    and returns a heatmap where each grid represents the output produced by
    convolution at each location. You can add optional parameters (e.g. stride,
    window_size, padding) to create additional functionality.
    '''
    (n_rows,n_cols,n_channels) = np.shape(I)

    # In the weakened version, use only the first channel
    if weak:
        n_channels = 1

    (box_height, box_width, n_channels) = np.shape(T)

    '''
    BEGIN YOUR CODE
    '''

    # heatmap will be padded with zeros to match size of original image
    heatmap = np.zeros((n_rows, n_cols))

    # Convert template channels into normalized vectors
    img_means = [np.mean(I[:,:,0]), np.mean(I[:,:,1]), np.mean(I[:,:,2])]
    img_stds = [np.std(I[:,:,0]), np.std(I[:,:,1]), np.std(I[:,:,2])]

    template_ch1 = T[:,:,0].flatten()
    template_ch1 = (template_ch1 - img_means[0]) / img_stds[0]
    template_ch1 = template_ch1 / np.linalg.norm(template_ch1)

    template_ch2 = T[:,:,1].flatten()
    template_ch2 = (template_ch2 - img_means[1]) / img_stds[1]
    template_ch2 = template_ch2 / np.linalg.norm(template_ch2)

    template_ch3 = T[:,:,2].flatten()
    template_ch3 = (template_ch3 - img_means[2]) / img_stds[2]
    template_ch3 = template_ch3 / np.linalg.norm(template_ch3)

    templates = [template_ch1, template_ch2, template_ch3]

    # upper_left will keep track of where we are in our scan
    upper_left = [0,0]

    while(True):
        # These if statements check to ensure our "sliding window" is in bounds
        if upper_left[1] >= n_cols-box_width:
            upper_left[1] = 0
            upper_left[0] += 1

        if upper_left[0] >= n_rows-box_height:
            break

        # Run matched filtering for each channel
        ip_avg = 0
        for n in range(n_channels):
            # Current "sliding window" section of the photo
            right_col = upper_left[0]+box_height
            bottom_row = upper_left[1]+box_width
            curr = I[upper_left[0]:right_col,
            upper_left[1]:bottom_row,n]

            # convert curr to normalized vector
            curr = curr.flatten()
            curr = (curr - img_means[n]) / img_stds[n]
            curr = curr / np.linalg.norm(curr)

            # Compute inner product
            ip = np.inner(curr, templates[n])
            ip_avg += ip

        # Add ip_avg to heatmap
        heatmap[upper_left[0], upper_left[1]] = ip_avg/n_channels

        upper_left[1] += 1

    '''
    END YOUR CODE
    '''

    return heatmap


def predict_boxes(heatmap, box_height, box_width, n_channels):
    '''
    This function takes heatmap and returns the bounding boxes and associated
    confidence scores.
    '''

    output = []

    '''
    BEGIN YOUR CODE

    As an example, here's code that generates between 1 and 5 random boxes
    of fixed size and returns the results in the proper format.

    num_boxes = np.random.randint(1,5)

    for i in range(num_boxes):
        (n_rows,n_cols,n_channels) = np.shape(I)

        tl_row = np.random.randint(n_rows - box_height)
        tl_col = np.random.randint(n_cols - box_width)
        br_row = tl_row + box_height
        br_col = tl_col + box_width

        score = np.random.random()

        output.append([tl_row,tl_col,br_row,br_col, score])

    '''
    threshold = 0.5

    # Scan the image using the template as our filter
    # In each "scan" take the maximum in the area of the heatmap currently
    # enclosed by our filter and put a bounding box if it passes a threshold
    upper_left = (0,0)

    # d will hold the maximum seen in every step of our scan.
    d = {}
    while True:
        for i in range(box_height):
            for j in range(box_width):
                ip_avg = 0
                row = upper_left[0] + i
                col = upper_left[1] + j

                if row >= heatmap.shape[0] or col >= heatmap.shape[1]:
                    break

                ip_avg = heatmap[row][col]

                if ip_avg > threshold:
                    # This was the best bounding box in this section of the heatmap
                    if (not upper_left in d) or ((upper_left in d) and ip_avg > d[upper_left][2]):
                        d[upper_left] = [row,col,ip_avg]

        if upper_left in d:
            # This current view of the heatmap contains a bounding box
            tl_row, tl_col, ip_avg = d[upper_left]
            br_row = tl_row+box_height
            br_col = tl_col+box_width

            # The confidence score will be a function of the inner product
            score = (ip_avg - threshold)/(1-threshold)

            output.append([tl_row,tl_col,br_row,br_col,score])

        # Our stride is equal to the box dimensions in this case
        upper_left = (upper_left[0], upper_left[1] + box_width)
        if upper_left[1] >= heatmap.shape[1]:
            # start new row
            upper_left = (upper_left[0] + box_height, 0)

        if upper_left[0] >= heatmap.shape[0]:
            # Finished our scan
            break

    '''
    END YOUR CODE
    '''

    return output


def detect_red_light_mf(I, weak=False):
    '''
    This function takes a numpy array <I> and returns a list <output>.
    The length of <output> is the number of bounding boxes predicted for <I>.
    Each entry of <output> is a list <[row_TL,col_TL,row_BR,col_BR,score]>.
    The first four entries are four integers specifying a bounding box
    (the row and column index of the top left corner and the row and column
    index of the bottom right corner).
    <score> is a confidence score ranging from 0 to 1.

    Note that PIL loads images in RGB order, so:
    I[:,:,0] is the red channel
    I[:,:,1] is the green channel
    I[:,:,2] is the blue channel
    '''

    '''
    BEGIN YOUR CODE
    '''
    # read in kernel using PIL:
    T = Image.open("kernel.jpg")

    # convert to numpy array:
    T = np.asarray(T)[:,:,0:3]
    box_height, box_width, n_channels = T.shape

    heatmap = compute_convolution(I, T, weak)
    output = predict_boxes(heatmap, box_height, box_width, n_channels)

    '''
    END YOUR CODE
    '''

    for i in range(len(output)):
        assert len(output[i]) == 5
        assert (output[i][4] >= 0.0) and (output[i][4] <= 1.0)

    return output, heatmap.tolist()

# Note that you are not allowed to use test data for training.
# set the path to the downloaded data:
data_path = '../data/RedLights2011_Medium'

# load splits:
split_path = '../data/hw02_splits'
file_names_train = np.load(os.path.join(split_path,'file_names_train.npy'))
file_names_test = np.load(os.path.join(split_path,'file_names_test.npy'))

# set a path for saving predictions:
preds_path = '../data/hw02_preds'
os.makedirs(preds_path, exist_ok=True) # create directory if needed

# Set this parameter to True when you're done with algorithm development:
done_tweaking = True
run_train = False

if run_train:
    '''
    Make predictions on the training set.
    '''
    preds_train = {}
    heatmaps_train = {}
    for i in tqdm(range(len(file_names_train))):

        # read image using PIL:
        I = Image.open(os.path.join(data_path,file_names_train[i]))

        # convert to numpy array:
        I = np.asarray(I)

        preds_train[file_names_train[i]], heatmaps_train[file_names_train[i]] = detect_red_light_mf(I, False)

    # save preds (overwrites any previous predictions!)
    with open(os.path.join(preds_path,'preds_train.json'),'w') as f:
        json.dump(preds_train,f)

    with open(os.path.join(preds_path,'heatmaps_train.json'),'w') as f:
        json.dump(heatmaps_train, f)

if done_tweaking:
    '''
    Make predictions on the test set.
    '''
    preds_test = {}
    heatmaps_test = {}
    for i in tqdm(range(len(file_names_test))):

        # read image using PIL:
        I = Image.open(os.path.join(data_path,file_names_test[i]))

        # convert to numpy array:
        I = np.asarray(I)

        preds_test[file_names_test[i]], heatmaps_test[file_names_test[i]] = detect_red_light_mf(I, False)

    # save preds (overwrites any previous predictions!)
    with open(os.path.join(preds_path,'preds_test.json'),'w') as f:
        json.dump(preds_test,f)

    with open(os.path.join(preds_path,'heatmaps_test.json'),'w') as f:
        json.dump(heatmaps_test, f)
