import lab1_1
import cv21_lab1_part2_utils
import cv2
import numpy as np
import matplotlib.pyplot as plt

# takes 2d numpy array (image) and returns
# an array of 3 columns: points and the corresponding sigma
#  (x, y, sigma)
def return_kp_data(img, sigma):
    # argwhere returns x coordinates as y and reverse
    # image x is horizontal while array x is vertical
    result = np.argwhere(img > 0)
    result[:, [0, 1]] = result[:, [1, 0]]
    result = np.insert(result, 2, sigma, axis=1)
    return result


# takes img((2d numpy array)), rho, k and returns criteria R(x, y)
def tanystis(img, sigma, rho, k):
    # two gaussians, rho and sigma specific.
    gaussian_sigma = lab1_1.get_gaussian(sigma)
    gaussian_rho = lab1_1.get_gaussian(rho)
    # be careful gradient is firts computed for rows and then for columns
    (gaus_y_der, gaus_x_der) = np.gradient(gaussian_sigma)
    i_s_x = lab1_1.convolution(img, gaus_x_der)
    i_s_y = lab1_1.convolution(img, gaus_y_der)
    j1 = lab1_1.convolution(i_s_x*i_s_x, gaussian_rho)
    j2 = lab1_1.convolution(i_s_x * i_s_y, gaussian_rho)
    j3 = lab1_1.convolution(i_s_y * i_s_y, gaussian_rho)

    l_plus = (j1 + j3 + np.sqrt(np.square(j1 - j3) + 4 * np.square(j2)))/2
    l_minus = (j1 + j3 - np.sqrt(np.square(j1 - j3) + 4 * np.square(j2)))/2
    # compute criteria r from l_minus and l_plus
    r = l_minus*l_plus-k*np.square(l_minus+l_plus)
    return r


# for a given r we apply the two asked conditions
def criteria(r, sigma, theta):
    ns = np.ceil(3*sigma)*2+1
    b_sq = cv21_lab1_part2_utils.disk_strel(ns)
    cond1 = (r == cv2.dilate(r, b_sq))
    cond2 = r > theta*np.max(r)
    return np.logical_and(cond1, cond2)

# returns well-chosen points for various scales
def multiscaled_metric(img, points, sigma, N):
    metrics = []

    # compute normalized log for all sigma
    for i in range(N):
        sigma_i = (s ** i) * sigma
        log_sigma = lab1_1.get_LoG(sigma_i)
        log = lab1_1.convolution(img, log_sigma)
        normalized_log = sigma_i ** 2 * np.abs(log)
        metrics.append(normalized_log)


    results = []
    for i in range(N):
        for (x, y, sigma) in points[i]:
            if N == 1:
                results.append((x, y, sigma))
            # be careful on the edges we need to compare only two scales
            elif i == 0:
                if metrics[i][y][x] >= metrics[i + 1][y][x]:
                    results.append((x, y, sigma))
            elif i == N - 1:
                if metrics[i][y][x] >= metrics[i - 1][y][x]:
                    results.append((x, y, sigma))
            else:
                if metrics[i][y][x] >= metrics[i - 1][y][x] and metrics[i][y][x] >= metrics[i + 1][y][x]:
                    results.append((x, y, sigma))
    int_points = np.array(results)
    return int_points


# compute the 3 tensors with given sigma, rho, theta and
# then apply the criteria to detect corners
def harrislap(img, s, N, sigma, rho, theta, k):
    points = []
    for i in range(N):
        sigma_i = s**i * sigma
        rho_i = s**i * rho

        r = tanystis(img, sigma_i, rho_i, k)
        cond = criteria(r, sigma_i, theta)
        res = return_kp_data(cond, sigma_i)
        points.append(res)

    res = multiscaled_metric(img, points, sigma, N)
    return res

# apply harris laplace in order to detect (x, y) where corners exist
# nothing is returns, corners are visualised with interest_points_visualization
def corners(image, sigma, rho, k, theta, s, N):
    img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    img = img.astype(np.float) / 255
    color_img = cv2.imread(image, cv2.IMREAD_COLOR)
    color_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)

    corner_points = harrislap(img, s, N, sigma, rho, theta, k)

    cv21_lab1_part2_utils.interest_points_visualization(color_img, corner_points)
    plt.show()
    return


# takes image and sigma, computes laplacian of gaussian
# convolutes image with them and returns det of hessian matrix
def find_hessian(img, sigma):
    gaus = lab1_1.get_gaussian(sigma)
    (g_y, g_x) = np.gradient(gaus)
    (g_xy, g_xx) = np.gradient(g_x)
    g_yy = np.gradient(g_y, axis=0)
    l_xx = lab1_1.convolution(img, g_xx)
    l_xy = lab1_1.convolution(img, g_xy)
    l_yy = lab1_1.convolution(img, g_yy)
    prod1 = l_xx * l_yy
    prod2 = l_xy * l_xy
    result = prod1 - prod2
    return result


# takes image, sigma, theta,  find_hessian is used to compute det in order to
# apply the asked criteria and find the blobs for one scale
def single_blobs(img, sigma, theta):
    hes_det = find_hessian(img, sigma)
    points = criteria(hes_det, sigma, theta)
    int_points = return_kp_data(points, sigma)
    return int_points

# detect multiscaled blobs
# takes image, sigma, theta, s, N, multiscaled_metric is used
# visualizes chosen points in various scales.

def blobs(image, sigma, theta, s, N):
    img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    img = img.astype(np.float) / 255
    color_img = cv2.imread(image, cv2.IMREAD_COLOR)
    color_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)

    points = []
    for i in range(N):
        sigma_i = s ** i * sigma
        pts = single_blobs(img, sigma_i, theta)
        points.append(pts)

    int_points = multiscaled_metric(img, points, sigma, N)
    cv21_lab1_part2_utils.interest_points_visualization(color_img, int_points)
    plt.show()
    return int_points


def calc_integral(img):
    img = np.cumsum(img, axis=0)
    img = np.cumsum(img, axis=1)
    return img


def box_metric(l_xx, l_yy, l_xy):
    r = l_xx * l_yy - (0.9 * l_xy) ** 2
    return r

# returns partial sum for a given box
def box_partial_sum(integral_img, x, y, pad_width):
    sh_x = int((x + 1)/2 - 1)
    sh_y = int((y + 1)/2 - 1)

    # shift filter corners to center of box
    upper_line = np.roll(integral_img, sh_y + 1, axis=0)
    lower_line = np.roll(integral_img, -sh_y, axis=0)

    up_right = np.roll(upper_line, -sh_x, axis=1)
    up_left = np.roll(upper_line, sh_x + 1, axis=1)

    down_right = np.roll(lower_line, -sh_x, axis=1)
    down_left = np.roll(lower_line, sh_x + 1, axis=1)

    # partial sum on integral image
    der_padded = down_right + up_left - down_left - up_right
    # remove padding
    der = der_padded[pad_width:-pad_width, pad_width:-pad_width]
    return der


def box_derrivatives(image, sigma, theta):
    img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    img = img.astype(np.float) / 255
    gaussian = lab1_1.get_gaussian(sigma)
    img = lab1_1.convolution(img, gaussian)

    n = 2 * np.ceil(3 * sigma) + 1
    short = int(2 * np.floor(n / 6) + 1)
    long = int(4 * np.floor(n / 6) + 1)

    pad_width = 2 * short
    padded = np.pad(img, pad_width, mode='reflect')
    integral_img = calc_integral(padded)

    L_xx = -3*(box_partial_sum(integral_img, short, long, pad_width)) + \
        box_partial_sum(integral_img, 3*short, long, pad_width)

    L_yy = -3*(box_partial_sum(integral_img, long, short, pad_width)) + \
        box_partial_sum(integral_img, long, 3*short, pad_width)

    box = box_partial_sum(integral_img, short, short, pad_width)
    padded = np.pad(box, pad_width, mode='reflect')
    sh_val = int((short-1)/2 + 1)
    ul = np.roll(padded, [sh_val, sh_val], axis=(0, 1))
    ur = np.roll(padded, [-sh_val, sh_val], axis=(0, 1))
    dl = np.roll(padded, [sh_val, -sh_val], axis=(0, 1))
    dr = np.roll(padded, [-sh_val, -sh_val], axis=(0, 1))
    L_xy_padded = ul + dr - ur - dl
    L_xy = L_xy_padded[pad_width:-pad_width, pad_width:-pad_width]

    r = box_metric(L_xx, L_yy, L_xy)
    points = criteria(r, sigma, theta)
    int_points = return_kp_data(points, sigma)
    return int_points


def box_filter_blobs(image, sigma, theta, s, N):
    img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    img = img.astype(np.float) / 255
    color_img = cv2.imread(image, cv2.IMREAD_COLOR)
    color_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)

    points = []
    for i in range(N):
        sigma_i = (s ** i) * sigma
        pts = box_derrivatives(image, sigma_i, theta)
        points.append(pts)

    int_points = multiscaled_metric(img, points, sigma, N)
    cv21_lab1_part2_utils.interest_points_visualization(color_img, int_points)
    plt.show()
    return int_points


if __name__ == '__main__':
    # sigma = 2, rho = 2.5, k = 0.05, theta = 0.005, s = 1.5, N = 4

    sigma = 2.5
    rho = 3
    k = 0.05
    theta = 0.005
    s = 1.5
    N = 1
    # images: urban_edges.jpg,  mars.png, blood_smear.jpg
    image = 'urban_edges.jpg'
    corners(image, sigma, rho, k, theta, s, N)



    sigma = 2.5
    rho = 3
    k = 0.1
    theta = 0.005
    s = 1.5
    N = 6
    image = 'blood_smear.jpg'
    corners(image, sigma, rho, k, theta, s, N)
    image = 'mars.png'
    corners(image, sigma, rho, k, theta, s, N)

    sigma = 2
    theta = 0.01
    s = 1.5
    N = 1
    image = 'blood_smear.jpg'
    blobs(image, sigma, theta, s, N)
    image = 'mars.png'
    blobs(image, sigma, theta, s, N)

    sigma = 2
    theta = 0.01
    s = 1.5
    N = 6
    image = 'blood_smear.jpg'
    blobs(image, sigma, theta, s, N)
    image = 'mars.png'
    blobs(image, sigma, theta, s, N)

    sigma = 2
    theta = 0.001
    s = 1.3
    N = 8
    image = 'blood_smear.jpg'
    box_filter_blobs(image, sigma, theta, s, N)
    image = 'mars.png'
    box_filter_blobs(image, sigma, theta, s, N)


