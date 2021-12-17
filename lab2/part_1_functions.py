import scipy.io
import scipy.stats
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.ndimage import label, map_coordinates


# find bounding boxes of image
def bounding_box(image, max_val, mu, cov):
    # convert image to y cb cr
    imgcbcr = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)[:, :, 1:]
    # get skin probability density
    frame = scipy.stats.multivariate_normal.pdf(imgcbcr, mean=mu, cov=cov)
    print(frame.shape)

    # normalize probability density function and threshold it to get binary probability
    frame_normalized = frame * max_val
    frame_threshold = (frame_normalized > 0.05).astype(float)
    plt.imshow(frame_threshold, cmap='gray')
    plt.show()

    # open with small element
    open_kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    opening = cv2.morphologyEx(frame_threshhold, cv2.MORPH_OPEN, open_kernel)
    plt.imshow(opening, cmap='gray')
    plt.show()
    # close with large element
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (53, 53))
    openclose = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, close_kernel)
    plt.imshow(openclose, cmap='gray')
    plt.show()

    # label each connected area
    labeled_array, num_features = label(openclose)
    head_vert, head_hor = np.where(labeled_array == 1)
    left_vert, left_hor = np.where(labeled_array == 2)
    right_vert, right_hor = np.where(labeled_array == 3)

    # find coordinates for each feature
    head_up = np.min(head_vert)
    head_down = np.max(head_vert)
    head_left = np.min(head_hor)
    head_right = np.max(head_hor)

    left_up = np.min(left_vert)
    left_down = np.max(left_vert)
    left_left = np.min(left_hor)
    left_right = np.max(left_hor)

    right_up = np.min(right_vert)
    right_down = np.max(right_vert)
    right_left = np.min(right_hor)
    right_right = np.max(right_hor)

    return ([head_left - 8, head_up, head_right - head_left + 16, head_down - head_up],
            [left_left - 16, left_up - 16, left_right - left_left + 32, left_down - left_up + 22],
            [right_left - 4, right_up - 4, right_right - right_left + 12, right_down - right_up + 12])


# lucas kanade algorithm implementation
def lk(I1, I2, features, rho, epsilon, d_x0, d_y0):
    I1 = I1.astype(np.float) / 255
    I2 = I2.astype(np.float) / 255
    n = int(np.ceil(3 * rho) * 2 + 1)
    gauss1D = cv2.getGaussianKernel(n, rho)
    gauss2D = gauss1D @ gauss1D.T
    # find spatial derivatives
    I1_y, I1_x = np.gradient(I1)

    threshold = 0.01
    result_matx = np.zeros(I1.shape)
    result_maty = np.zeros(I1.shape)
    features_moved_x = []
    features_moved_y = []

    # repeat for every feature
    for feature in features:
        (x, y) = (feature[0].astype(int)[0], feature[0].astype(int)[1])

        # crop around each feature
        left_y_edge = max(0, y - (n - 1) // 2)
        right_y_edge = min(y + (n - 1) // 2, I1.shape[0])
        left_x_edge = max(0, x - (n - 1) // 2)
        right_x_edge = min(x + (n - 1) // 2, I1.shape[1])

        I_feat_1 = I1[left_y_edge:right_y_edge, left_x_edge:right_x_edge]
        I_feat_x = I1_x[left_y_edge:right_y_edge, left_x_edge:right_x_edge]
        I_feat_y = I1_y[left_y_edge:right_y_edge, left_x_edge:right_x_edge]

        x_index, y_index = np.meshgrid(np.arange(I_feat_1.shape[1]), np.arange(I_feat_1.shape[0]))
        d_x0_feat = d_x0[left_y_edge:right_y_edge, left_x_edge:right_x_edge]
        d_y0_feat = d_y0[left_y_edge:right_y_edge, left_x_edge:right_x_edge]
        dx, dy = d_x0_feat, d_y0_feat

        # repeat calculations until error decreases or iterations > 900
        for i in range(900):
            Iprev = map_coordinates(I_feat_1, [np.ravel(y_index + d_y0_feat), np.ravel(x_index + d_x0_feat)], order=1) \
                .reshape(I_feat_1.shape[0], I_feat_1.shape[1])
            A1 = map_coordinates(I_feat_x, [np.ravel(y_index + d_y0_feat), np.ravel(x_index + d_x0_feat)], order=1) \
                .reshape(I_feat_x.shape[0], I_feat_x.shape[1])
            A2 = map_coordinates(I_feat_y, [np.ravel(y_index + d_y0_feat), np.ravel(x_index + d_x0_feat)], order=1) \
                .reshape(I_feat_y.shape[0], I_feat_y.shape[1])

            E = I2[left_y_edge:right_y_edge, left_x_edge:right_x_edge] - Iprev

            a11 = cv2.filter2D(A1 ** 2, -1, gauss2D) + epsilon
            a12 = cv2.filter2D(A1 * A2, -1, gauss2D)
            a22 = cv2.filter2D(A2 ** 2, -1, gauss2D) + epsilon
            b1 = cv2.filter2D(A1 * E, -1, gauss2D)
            b2 = cv2.filter2D(A2 * E, -1, gauss2D)
            det = a11 * a22 - a12 * a12
            u_x = (a22 * b1 - a12 * b2) / det
            u_y = (a11 * b2 - a12 * b1) / det
            dx += u_x
            dy += u_y
            if np.linalg.norm(u_x) < threshold and np.linalg.norm(u_y) < threshold:
                break

        result_matx[y, x] = dx[(min(y + n // 2, I1.shape[0]) - max(0, y - n // 2)) // 2][
            (min(x + n // 2, I1.shape[1]) - max(0, x - n // 2)) // 2]
        result_maty[y, x] = dy[(min(y + n // 2, I1.shape[0]) - max(0, y - n // 2)) // 2][
            (min(x + n // 2, I1.shape[1]) - max(0, x - n // 2)) // 2]
        features_moved_x.append(result_matx[y, x])
        features_moved_y.append(result_maty[y, x])
    return result_matx, result_maty, features_moved_x, features_moved_y


# gaussian filtering and subsampling
def GREDUCE(I):
    gauss1D = cv2.getGaussianKernel(19, 3)
    gauss2D = gauss1D @ gauss1D.T
    convoluted_image = cv2.filter2D(I, -1, gauss2D)
    return convoluted_image[0::2, 0::2]


# create gaussian pyramid of given depth
def GPyramid(I, depth):
    g_pyr = [I]
    for i in range(depth - 1):
        g_pyr.append(GREDUCE(g_pyr[i]))
    g_pyr.reverse()
    return g_pyr


def displ(arr1, arr2, thres=0.1):
    opt_vect = []

    for dx, dy in zip(arr1, arr2):
        E = dx**2 + dy**2
        if E <= thres:
            continue
        opt_vect.append(np.array([dx, dy]))

    if len(opt_vect) == 0:
        return [0.0, 0.0]

    opt_vect = np.array(opt_vect)
    # Define Vectors #
    dxs = opt_vect[:, 0]
    dys = opt_vect[:, 1]

    return [np.mean(dxs), np.mean(dys)]


def lk_Gpyramid(I1, I2, rho, epsilon, depth, maxCorners):
    Gp_1 = GPyramid(I1, depth)
    Gp_2 = GPyramid(I2, depth)
    d_x0 = np.zeros(Gp_1[0].shape)
    d_y0 = np.zeros(Gp_1[0].shape)
    [displ_x, displ_y] = [0, 0]
    [resx, resy] = [0, 0]
    for i in range(depth):
        blockSize = 7
        feature_params = dict(maxCorners=maxCorners, qualityLevel=0.001, minDistance=7, blockSize=blockSize)
        features = cv2.goodFeaturesToTrack(Gp_2[i], mask=None, **feature_params)
        while features is None and blockSize > 2:
            blockSize -= 2
            feature_params = dict(maxCorners=maxCorners, qualityLevel=0.001, minDistance=7, blockSize=blockSize)
            features = cv2.goodFeaturesToTrack(Gp_2[i], mask=None, **feature_params)
        if blockSize < 2:
            [displ_x, displ_y] = [0, 0]
        else:
            resx, resy, features_moved_x, features_moved_y = lk(Gp_1[i], Gp_2[i], features, rho, epsilon, d_x0, d_y0)
            [displ_x, displ_y] = displ(features_moved_x, features_moved_y)
        if i < depth - 1:
            d_x0 = np.full(Gp_1[i + 1].shape, displ_x * 2)
            d_y0 = np.full(Gp_1[i + 1].shape, displ_y * 2)

    return displ_x, displ_y, resx, resy


def display_flow(resx, resy):
    plt.quiver(-resx, -resy, angles='xy', width=0.003, color='blue', scale=100)
    plt.gca().invert_yaxis()
    plt.show()
