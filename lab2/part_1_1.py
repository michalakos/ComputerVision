import scipy.io
import scipy.stats
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.ndimage import label, map_coordinates
import ask1_functions as ask1

# load skinSamples from given mat file
mat = scipy.io.loadmat('GreekSignLanguage/skinSamplesRGB.mat')
skinSamplesRGB = mat['skinSamplesRGB']
skinSamplesRGB = np.reshape(skinSamplesRGB, (1, 1782, 3))
imgYCC = cv2.cvtColor(skinSamplesRGB, cv2.COLOR_RGB2YCrCb)[0, :, 1:]

# keep only Cr and Cb channels, because Y is for brightness, so we don't care about it now
Cr = imgYCC[:, 0]
Cb = imgYCC[:, 1]

# find mean and covariance values of skin samples in order to categorize skin color and
# predict skin areas in other images
cr_mean = np.mean(Cr)
cb_mean = np.mean(Cb)
mean_vect = [cb_mean, cr_mean]

cov_matrix = np.cov(np.transpose(imgYCC))
max_val = np.sqrt(np.linalg.det(cov_matrix)) * 2 * 3.14

print("Mean values = ", mean_vect)
print("Covariace matrix = ", cov_matrix)

# read first frame and move it by 3 pixels
img1 = cv2.imread('GreekSignLanguage/1.png', cv2.IMREAD_COLOR)
test_image = np.roll(img1, 3, axis=1)
# convert images to grayscale
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
test_gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)

# found bounding boxes for first image, that we'll use later on
(head_box, left_box, right_box) = ask1.bounding_box(img1, max_val, mean_vect, cov_matrix)
print("head box = ", head_box)
print("left_box = ", left_box)
print("right_box = ", right_box)

# slice image to get bounding boxes around every feature
head_bb_1 = gray1[head_box[1]:head_box[1] + head_box[3], head_box[0]:head_box[0] + head_box[2]]
left_bb_1 = gray1[left_box[1]:left_box[1] + left_box[3], left_box[0]:left_box[0] + left_box[2]]
right_bb_1 = gray1[right_box[1]:right_box[1] + right_box[3], right_box[0]:right_box[0] + right_box[2]]

# Display first frame with the calculated bounding boxes
fig, ax = plt.subplots()
ax.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
rect = patches.Rectangle((head_box[0], head_box[1]), head_box[2], head_box[3], linewidth=1, edgecolor='r',
                         facecolor='none')
rect2 = patches.Rectangle((left_box[0], left_box[1]), left_box[2], left_box[3], linewidth=1, edgecolor='g',
                          facecolor='none')
rect3 = patches.Rectangle((right_box[0], right_box[1]), right_box[2], right_box[3], linewidth=1, edgecolor='b',
                          facecolor='none')
ax.add_patch(rect)
ax.add_patch(rect2)
ax.add_patch(rect3)
plt.show()

# repeat for moved image
test_head_bb = test_gray[head_box[1]:head_box[1] + head_box[3], head_box[0]:head_box[0] + head_box[2]]
test_left_bb = test_gray[left_box[1]:left_box[1] + left_box[3], left_box[0]:left_box[0] + left_box[2]]
test_right_bb = test_gray[right_box[1]:right_box[1] + right_box[3], right_box[0]:right_box[0] + right_box[2]]

head_displ_x, head_displ_y, head_bb_moved_x, head_bb_moved_y = ask1.lk_Gpyramid(head_bb_1, test_head_bb, 4, 0.01, 3, 17)
print("test head = ", head_displ_x, head_displ_y)
left_displ_x, left_displ_y, left_bb_moved_x, left_bb_moved_y = ask1.lk_Gpyramid(left_bb_1, test_left_bb, 4, 0.01, 3, 15)
print("test left = ", left_displ_x, left_displ_y)
right_displ_x, right_displ_y, right_bb_moved_x, right_bb_moved_y = ask1.lk_Gpyramid(right_bb_1, test_right_bb, 4, 0.01, 3, 15)
print("test right = ", right_displ_x, right_displ_y)

head_box = [head_box[0] - int(np.round(head_displ_x)), head_box[1] - int(np.round(head_displ_y)), head_box[2],
            head_box[3]]
left_box = [left_box[0] - int(np.round(left_displ_x)), left_box[1] - int(np.round(left_displ_y)), left_box[2],
            left_box[3]]
right_box = [right_box[0] - int(np.round(right_displ_x)), right_box[1] - int(np.round(right_displ_y)), right_box[2],
             right_box[3]]

# display second frame with the calculated bounding boxes
fig, ax = plt.subplots()
ax.imshow(cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB))
rect = patches.Rectangle((head_box[0], head_box[1]), head_box[2], head_box[3], linewidth=1, edgecolor='r',
                         facecolor='none')
rect2 = patches.Rectangle((left_box[0], left_box[1]), left_box[2], left_box[3], linewidth=1, edgecolor='g',
                          facecolor='none')
rect3 = patches.Rectangle((right_box[0], right_box[1]), right_box[2], right_box[3], linewidth=1, edgecolor='b',
                          facecolor='none')
ax.add_patch(rect)
ax.add_patch(rect2)
ax.add_patch(rect3)
plt.show()

# find features for head
feature_params = dict(maxCorners=17, qualityLevel=0.001, minDistance=5, blockSize=5)
test_head_features = cv2.goodFeaturesToTrack(test_head_bb, mask=None, **feature_params)
plt.imshow(test_head_bb, cmap='gray')
plt.plot(test_head_features[:, 0, 0], test_head_features[:, 0, 1], color='green', marker='o', linestyle='None',
         markersize=4)
# plt.savefig("./first_test_plots/" + "head_feats.png")
plt.show()
plt.clf()
# display movement of each feature in head bounding box
plt.quiver(-head_bb_moved_x, -head_bb_moved_y, angles='xy', width=0.003, scale=1 / 0.02)
plt.gca().invert_yaxis()
# plt.savefig("./first_test_plots/" + "head_flow.png")
plt.show()
plt.clf()

# repeat for left hand
feature_params = dict(maxCorners=15, qualityLevel=0.001, minDistance=5, blockSize=7)
test_left_features = cv2.goodFeaturesToTrack(test_left_bb, mask=None, **feature_params)
plt.imshow(test_left_bb, cmap='gray')
plt.plot(test_left_features[:, 0, 0], test_left_features[:, 0, 1], color='green', marker='o', linestyle='None',
         markersize=4)
# plt.savefig("./first_test_plots/" + "left_feats.png")
plt.show()
plt.clf()
plt.quiver(-left_bb_moved_x, -left_bb_moved_y, angles='xy', width=0.003, scale=1 / 0.02)
plt.gca().invert_yaxis()
# plt.savefig("./first_test_plots/" + "left_flow.png")
plt.show()
plt.clf()

# repeat for right hand
feature_params = dict(maxCorners=15, qualityLevel=0.001, minDistance=5, blockSize=7)
test_right_features = cv2.goodFeaturesToTrack(test_right_bb, mask=None, **feature_params)
plt.imshow(test_right_bb, cmap='gray')
plt.plot(test_right_features[:, 0, 0], test_right_features[:, 0, 1], color='green', marker='o', linestyle='None',
         markersize=4)
# plt.savefig("./first_test_plots/" + "right_feats.png")
plt.show()
plt.clf()
plt.quiver(-right_bb_moved_x, -right_bb_moved_y, angles='xy', width=0.003, scale=1 / 0.02)
plt.gca().invert_yaxis()
# plt.savefig("./first_test_plots/" + "right_flow.png")
plt.show()
plt.clf()
