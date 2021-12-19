import scipy.io
import scipy.stats
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.ndimage import label, map_coordinates
import ask1_functions as ask1


output_flow_dir = './flow_plots/'
test_output = './diff_RE_flow/'
output_box_dir = './box_plots/'
οutput_box_test = './box_test_plots/'
output_multi_flow = './multi_flow_plots/'
output_multi_box = './multi_box_test/'

# read first two frames
img1 = cv2.imread('GreekSignLanguage/1.png', cv2.IMREAD_COLOR)
img2 = cv2.imread('GreekSignLanguage/2.png', cv2.IMREAD_COLOR)

gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# values found in previous step, so we use those values directly in order
# to not waste time repeating calculations
# (head_box, left_box, right_box) = ask1.bounding_box(img1, max_val, mean_vect, cov_matrix)
(head_box, left_box, right_box) = ([130, 88, 84, 125], [47, 243, 71, 66], [162, 265, 87, 49])

# slice bounding boxes of each feature of the image (head, left hand, right hand)
# for the first two frames
head_bb_1 = gray1[head_box[1]:head_box[1] + head_box[3], head_box[0]:head_box[0] + head_box[2]]
left_bb_1 = gray1[left_box[1]:left_box[1] + left_box[3], left_box[0]:left_box[0] + left_box[2]]
right_bb_1 = gray1[right_box[1]:right_box[1] + right_box[3], right_box[0]:right_box[0] + right_box[2]]

head_bb_2 = gray2[head_box[1]:head_box[1] + head_box[3], head_box[0]:head_box[0] + head_box[2]]
left_bb_2 = gray2[left_box[1]:left_box[1] + left_box[3], left_box[0]:left_box[0] + left_box[2]]
right_bb_2 = gray2[right_box[1]:right_box[1] + right_box[3], right_box[0]:right_box[0] + right_box[2]]


rho = 4
e = 0.05
head_displ_x, head_displ_y, head_bb_moved_x, head_bb_moved_y = ask1.lk_Gpyramid(head_bb_1, head_bb_2, rho, e, 3, 13)
left_displ_x, left_displ_y, left_bb_moved_x, left_bb_moved_y = ask1.lk_Gpyramid(left_bb_1, left_bb_2, rho, e, 3, 15)
right_displ_x, right_displ_y, right_bb_moved_x, right_bb_moved_y = ask1.lk_Gpyramid(right_bb_1, right_bb_2, rho, e, 3, 10)

print("head_displ = ", head_displ_x, head_displ_y)
print("left_displ = ", left_displ_x, left_displ_y)
print("right_displ = ", right_displ_x, right_displ_y)

ask1.display_flow(head_bb_moved_x, head_bb_moved_y, test_output + "head_flow_" + str(1))
ask1.display_flow(left_bb_moved_x, left_bb_moved_y, output_multi_flow + "left_flow_" + str(rho) + "_" + str(e))
ask1.display_flow(right_bb_moved_x, right_bb_moved_y, test_output + "right_flow_" + str(1))

# repeat for all images in dataset
for i in range(2, 66):
    img1 = img2.copy()
    img2 = cv2.imread('GreekSignLanguage/' + str(i + 1) + '.png', cv2.IMREAD_COLOR)
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    head_box = [head_box[0] - int(np.round(head_displ_x)), head_box[1] - int(np.round(head_displ_y)), head_box[2],
                head_box[3]]

    left_box = [left_box[0] - int(np.round(left_displ_x)), left_box[1] - int(np.round(left_displ_y)), left_box[2],
                left_box[3]]

    right_box = [right_box[0] - int(np.round(right_displ_x)), right_box[1] - int(np.round(right_displ_y)), right_box[2],
                 right_box[3]]

    head_bb_1 = gray1[head_box[1]:head_box[1] + head_box[3], head_box[0]:head_box[0] + head_box[2]]
    left_bb_1 = gray1[left_box[1]:left_box[1] + left_box[3], left_box[0]:left_box[0] + left_box[2]]
    right_bb_1 = gray1[right_box[1]:right_box[1] + right_box[3], right_box[0]:right_box[0] + right_box[2]]

    head_bb_2 = gray2[head_box[1]:head_box[1] + head_box[3], head_box[0]:head_box[0] + head_box[2]]
    left_bb_2 = gray2[left_box[1]:left_box[1] + left_box[3], left_box[0]:left_box[0] + left_box[2]]
    right_bb_2 = gray2[right_box[1]:right_box[1] + right_box[3], right_box[0]:right_box[0] + right_box[2]]

    # Create figure and axes
    fig, ax = plt.subplots(frameon=False)
    # Display the image
    ax.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    # Create a Rectangle patch
    rect = patches.Rectangle((head_box[0], head_box[1]), head_box[2], head_box[3], linewidth=1, edgecolor='r',
                             facecolor='none')
    rect2 = patches.Rectangle((left_box[0], left_box[1]), left_box[2], left_box[3], linewidth=1, edgecolor='g',
                              facecolor='none')
    rect3 = patches.Rectangle((right_box[0], right_box[1]), right_box[2], right_box[3], linewidth=1, edgecolor='b',
                              facecolor='none')

    # Add the patch to the Axes
    ax.set_axis_off()
    ax.add_patch(rect)
    ax.add_patch(rect2)
    ax.add_patch(rect3)

    plt.show()
    # used to save figure in local storage
    #plt.savefig(output_multi_box + "box_4_005_" + str(i)+'.png',  bbox_inches='tight', pad_inches=0)
    plt.clf()
    plt.close(fig)

    head_displ_x, head_displ_y, head_bb_moved_x, head_bb_moved_y = ask1.lk_Gpyramid(head_bb_1, head_bb_2, rho, e, 3, 13)
    left_displ_x, left_displ_y, left_bb_moved_x, left_bb_moved_y = ask1.lk_Gpyramid(left_bb_1, left_bb_2, rho, e, 3, 15)
    right_displ_x, right_displ_y, right_bb_moved_x, right_bb_moved_y = ask1.lk_Gpyramid(right_bb_1, right_bb_2, rho, e, 3, 10)

    ask1.display_flow(head_bb_moved_x, head_bb_moved_y, output_flow_dir + "head_flow_" + str(i))
    ask1.display_flow(left_bb_moved_x, left_bb_moved_y, output_flow_dir + "left_flow_" + str(i))
    ask1.display_flow(right_bb_moved_x, right_bb_moved_y, output_flow_dir + "right_flow_" + str(i))