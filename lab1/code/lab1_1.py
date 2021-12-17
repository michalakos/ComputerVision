import cv2
import numpy as np
import matplotlib.pyplot as plt


# takes image, mean and psnr value and returns the corresponding noisy image
# (2d numpy array)
def noise(img, mean, psnr):
    (rows, cols) = img.shape
    # x max and y max for image
    i_min = img.min()
    i_max = img.max()
    # find sigma for a given psnr
    sigma = (i_max-i_min)/(10**(psnr/20))

    # random.normal is used to create white noise with mean, sigma parameters
    white_noise_img = np.random.normal(mean, sigma, (rows, cols))
    white_noise_img = white_noise_img.reshape(rows, cols)
    noisy = img + white_noise_img
    return noisy


# returns 2d gaussian (numpy array) for a given sigma
def get_gaussian(sigma, plot=False):
    n = int(np.ceil(3*sigma)*2+1)
    gauss1D = cv2.getGaussianKernel(n, sigma)
    # 2d gaussian equals to matrix multiplication of 1d gaussian with the equivalent reversed
    gauss2D = gauss1D @ gauss1D.T
    # plot gaussian
    if plot:
        hf = plt.figure()
        ax = hf.add_subplot(111, projection='3d')

        # define x, y axis
        X = np.arange(-(n-1)/2, (n-1)/2+1, 1)
        Y = np.arange(-(n-1)/2, (n-1)/2+1, 1)
        X, Y = np.meshgrid(X, Y)
        # z is 2d gaussian function
        Z = gauss2D
        ax.plot_surface(X, Y, Z)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title('Gaussian for sigma = '+str(sigma))
        plt.show()
    return gauss2D

# returns Laplacian of Gaussian for a given sigma
# 2d numpy array
def get_LoG(sigma, plot=False):
    n = int(np.ceil(3*sigma)*2+1)
    # define kernel for LoG
    # (0,0) -> center of array
    # (abs(n-1)/2) -> edges of array
    vect = np.linspace(-(n-1)/2, (n-1)/2, n)
    [xs, ys] = np.meshgrid(vect, vect)
    # compute LoG with known mathematical type
    nom = (np.square(xs) + np.square(ys) - 2*(sigma**2))
    den = 2*np.pi*(sigma**6)
    exp = np.exp(-(np.square(xs)+np.square(ys))/(2*(sigma**2)))
    LoG = (nom/den)*exp
    # plot gaussian
    if plot:
        hf = plt.figure()
        ax = hf.add_subplot(111, projection='3d')
        X = np.arange(-(n - 1) / 2, (n - 1) / 2 + 1, 1)
        Y = np.arange(-(n - 1) / 2, (n - 1) / 2 + 1, 1)
        X, Y = np.meshgrid(X, Y)
        Z = LoG
        ax.plot_surface(X, Y, Z)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title('LoG for sigma = ' + str(sigma))
        plt.show()
    return LoG

# returns convoluted image for a given kernel
def convolution(img, kernel):
    result = cv2.filter2D(img, -1, kernel)
    return result

# takes image read with cv2 and
# returns 2d numpy array, non linear laplacian of image
def non_linear_laplacian(img):
    # define kernel for morphological filters
    kern = np.array([
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0]
    ], dtype=np.uint8)
    dilated_img = cv2.dilate(img, kern)
    eroded_img = cv2.erode(img, kern)
    non_lin = dilated_img + eroded_img - 2*img
    return non_lin

# takes image, converts it to binary  and returns
# dilated minus eroded image which is a
# 2d numpy array
def zero_crossings(img):
    binary = (img >= 0).astype(np.float)
    kern = np.array([
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0]
    ], dtype=np.uint8)
    dilated_img = cv2.dilate(binary, kern)
    eroded_img = cv2.erode(binary, kern)
    frontier = dilated_img - eroded_img
    return frontier


def smooth_decline(img, theta):
    (xs, ys) = np.gradient(img)
    gradient = np.sqrt(np.square(xs) + np.square(ys))
    I_max = gradient.max()
    result = (gradient > theta * I_max)
    return result


def edge_detection(img, sigma, theta, linear):
    gaussian_kernel = get_gaussian(sigma)
    gaussian = convolution(img, gaussian_kernel)

    if linear:
        log_kernel = get_LoG(sigma)
        laplacian = convolution(img, log_kernel)

    else:
        laplacian = non_linear_laplacian(gaussian)

    crossings = zero_crossings(laplacian)
    (xs, ys) = np.gradient(gaussian)
    grad = np.sqrt(np.square(xs)+np.square(ys))
    I_max = grad.max()
    return ((grad > theta*I_max) & (crossings == 1.0)).astype(np.float)

# returns real edges of image without noise
# dilation - erosion
def real_edges(img, theta):
    kern = np.array([
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0]
    ], dtype=np.uint8)
    dilated_img = cv2.dilate(img, kern)
    eroded_img = cv2.erode(img, kern)
    frontier = dilated_img - eroded_img
    return frontier > theta

# returns rating for our edge detection based on real edges of image.
def rating(found, real):
    # D and T (where D set of edges found from us and T set of real edges)
    intersection = found.astype(bool) & real.astype(bool)
    inter_sum = intersection.sum() # card(D and T)
    found_sum = found.sum() # card(D)
    real_sum = real.sum() # card(T)
    small = np.exp(-10)
    prdt = inter_sum/(real_sum+small)
    prtd = inter_sum/(found_sum+small)
    return (prdt+prtd)/2

def search_best_rating(noisy, linear, real_edge):
    sigma_end = 10.0
    sigma_step = 0.1
    theta_end = 1.0
    theta_step = 0.01
    best_rating = sigma_i = theta_i = 0
    for sigma in np.arange(0.0000000000000000000001, sigma_end, sigma_step):
        for theta in np.arange(0.0000000000000000000001, theta_end, theta_step):
            new_rating = rating(edge_detection(noisy, sigma, theta, linear), real_edge)
            if new_rating > best_rating:
                best_rating = new_rating
                sigma_i = sigma
                theta_i = theta
    print(best_rating, sigma_i, theta_i)
    edge = edge_detection(noisy, sigma_i, theta_i, True)
    plt.imshow(edge, cmap='gray')
    plt.show()


if __name__ == '__main__':

    image = 'edgetest_10.png'
    # read image
    img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    img = img.astype(np.float) / 255

    # display the image
    plt.imshow(img, cmap='gray')
    plt.show()

    # put noise in image
    noisy_20 = noise(img, 0, 20)
    plt.imshow(noisy_20, cmap='gray')
    plt.show()
    noisy_10 = noise(img, 0, 10)
    plt.imshow(noisy_10, cmap='gray')
    plt.show()

    
    # find and show real edges
    real_edge = real_edges(img, 0.15)
    plt.imshow(real_edge, cmap='gray')
    plt.show()



    # find best edges for linear laplacian of image
    # for PSNR=20dB
    search_best_rating(noisy_20, True, real_edge)
    '''
    # for PSNR=10dB
    search_best_rating(noisy_10, True, real_edge)

    # find best edges for non linear laplacian of image
    # for PSNR=20dB
    search_best_rating(noisy_20, False, real_edge)

    # for PSNR=10dB
    search_best_rating(noisy_10, False, real_edge)
    '''
    image = 'coffee.jpg'
    # read image
    img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    img = img.astype(np.float) / 255
    color_img = cv2.imread(image, cv2.IMREAD_COLOR)
    color_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
    # display the image
    #plt.imshow(color_img)
    #plt.show()
    # edge detection
    edge = edge_detection(img, 2, 0.015, True)
    plt.imshow(edge, cmap='gray')
    plt.show()
