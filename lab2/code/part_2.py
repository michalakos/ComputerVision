import glob
from cv21_lab2_2_utils import read_video, show_detection, orientation_histogram, bag_of_words, svm_train_test
import numpy as np
import cv2
import matplotlib.pyplot as plt
import scipy.ndimage as scp
import pickle

path = 'cv21_lab2_part2_material/data/'


# detector using harris method
def harris_3d(video, sigma, s, tau, k):
    video = video.astype(float) / 255

    n = int(np.ceil(3 * sigma) * 2 + 1)
    k1 = np.transpose(cv2.getGaussianKernel(n, sigma))[0]
    k2 = k1  # the kernel along the 2nd dimension
    n = int(np.ceil(3 * tau) * 2 + 1)
    k3 = np.transpose(cv2.getGaussianKernel(n, tau))[0]  # the kernel along the 3rd dimension
    # Convolve over all three axes to smooth out video
    smooth_video = video.copy()
    for i, kernel in enumerate((k1, k2, k3)):
        smooth_video = scp.convolve1d(smooth_video, kernel, axis=i)

    Ly, Lx, Lt = np.gradient(smooth_video)

    n = int(np.ceil(3 * s*sigma) * 2 + 1)
    k1 = np.transpose(cv2.getGaussianKernel(n, s*sigma))[0]
    k2 = k1  # the kernel along the 2nd dimension
    n = int(np.ceil(3 * s*tau) * 2 + 1)
    k3 = np.transpose(cv2.getGaussianKernel(n, s*tau))[0]  # the kernel along the 3rd dimension

    a11 = Lx*Lx
    for i, kernel in enumerate((k1, k2, k3)):
        a11 = scp.convolve1d(a11, kernel, axis=i)
    a12 = Lx*Ly
    for i, kernel in enumerate((k1, k2, k3)):
        a12 = scp.convolve1d(a12, kernel, axis=i)
    a13 = Lx*Lt
    for i, kernel in enumerate((k1, k2, k3)):
        a13 = scp.convolve1d(a13, kernel, axis=i)
    a22 = Ly*Ly
    for i, kernel in enumerate((k1, k2, k3)):
        a22 = scp.convolve1d(a22, kernel, axis=i)
    a23 = Ly*Lt
    for i, kernel in enumerate((k1, k2, k3)):
        a23 = scp.convolve1d(a23, kernel, axis=i)
    a33 = Lt*Lt
    for i, kernel in enumerate((k1, k2, k3)):
        a33 = scp.convolve1d(a33, kernel, axis=i)

    detM = a11*a22*a33 + 2*a12*a23*a13 - a11*a23*a23 - a12*a12*a33 - a13*a13*a22
    traceM = a11 + a22 + a33
    H_harris = np.abs(detM - k * (traceM * traceM * traceM))
    return H_harris


# detector using gabor filter method
def gabor(video, sigma, tau):
    video = video.astype(float) / 255

    n = int(np.ceil(3 * sigma) * 2 + 1)
    k1 = np.transpose(cv2.getGaussianKernel(n, sigma))[0]
    k2 = k1
    # the two spatial kernels are the same
    # Convolve over all x, y axes to smooth out video
    smooth_video = video.copy()
    for i, kernel in enumerate((k1, k2)):
        smooth_video = scp.convolve1d(smooth_video, kernel, axis=i)

    omega = 4/tau
    window = np.linspace(int(np.round(-2*tau)), int(np.round(2*tau)), 2*int(np.round(2*tau)) + 1, dtype=int)
    h_ev = np.cos(2 * np.pi * window * omega) * np.exp((- window**2) / (2 * tau**2))
    h_ev /= np.linalg.norm(h_ev, ord=1)
    h_od = np.sin(2 * np.pi * window * omega) * np.exp((- window**2) / (2 * tau**2))
    h_od /= np.linalg.norm(h_ev, ord=1)
    H_gabor = (scp.convolve1d(smooth_video, h_ev, axis=2))**2 + (scp.convolve1d(smooth_video, h_od, axis=2))**2
    return H_gabor


# given an interest intensity array returns N points with greatest values
def interest_points_extraction(H, N, sigma):
    flat = np.ndarray.flatten(H)
    asc = np.argsort(flat)
    desc = np.flip(asc)

    points = []
    for i in range(min(N, desc.shape[0])):
        (y, x, t) = np.unravel_index(desc[i], shape=H.shape)
        points.append((x, y, t, sigma))
    points = np.array(points)
    return points


# returns HoG descriptor for all int_points of a video
def get_HoG(video, int_points, sigma, nbins):
    der_y, der_x, der_t = np.gradient(video)
    side = int(np.round(4 * sigma))

    descriptors = []
    for point in int_points:
        leftmost = max(0, point[0] - side)
        rightmost = min(video.shape[1] - 1, point[0] + side + 1)
        uppermost = max(0, point[1] - side)
        lowermost = min(video.shape[0] - 1, point[1] + side + 1)

        descriptor = orientation_histogram(der_x[uppermost:lowermost, leftmost:rightmost, point[2]],
                                           der_y[uppermost:lowermost, leftmost:rightmost, point[2]],
                                           nbins, np.array([side, side]))
        descriptors.append(descriptor)

    return np.array(descriptors)


# returns HoF descriptor for all int_points of a video
def get_HoF(video, int_points, sigma, nbins):
    side = int(np.round(4 * sigma))
    oflow = cv2.DualTVL1OpticalFlow_create(nscales=1)

    descriptors = []
    for interest_point in int_points:
        leftmost = max(0, interest_point[0] - side)
        rightmost = min(video.shape[1] - 1, interest_point[0] + side + 1)
        uppermost = max(0, interest_point[1] - side)
        lowermost = min(video.shape[0] - 1, interest_point[1] + side + 1)
        if interest_point[2] == video.shape[2] - 1:
            interest_point[2] -= 1

        flow = oflow.calc(video[uppermost:lowermost, leftmost:rightmost, interest_point[2]],
                          video[uppermost:lowermost, leftmost:rightmost, interest_point[2] + 1], None)

        u_flow = flow[:, :, 0]
        v_flow = flow[:, :, 1]
        descriptor = orientation_histogram(u_flow, v_flow, nbins, np.array([side, side]))
        descriptors.append(descriptor)

    return np.array(descriptors)


# returns HoG/HoF descriptor for all int_points of a video
def get_HoG_HoF(video, int_points, sigma, nbins):
    hog_descs = get_HoG(video, int_points, sigma, nbins)
    hof_descs = get_HoF(video, int_points, sigma, nbins)
    new_descs = np.concatenate((hog_descs, hof_descs))
    return new_descs


# given the names of the videos in the training set returns the names of the videos in each set,
# as well as their labels
def train_test():
    fd = open('cv21_lab2_part2_material/data/training_videos.txt', 'r')
    file = fd.readline()
    train_videos = []

    all_videos = glob.glob(path+'running/*')
    all_videos += glob.glob(path+'boxing/*')
    all_videos += glob.glob(path+'walking/*')

    while file:
        file = file.replace('\n', '')
        if 'running' in file:
            label = 'running'
        elif 'boxing' in file:
            label = 'boxing'
        else:
            label = 'walking'
        video = path + label + '/' + file
        train_videos.append(video)
        file = fd.readline()

    test_videos = []
    for vid in all_videos:
        if vid not in train_videos:
            test_videos.append(vid)

    train_labels = []
    for i in range(len(train_videos)):
        if 'running' in train_videos[i]:
            train_labels.append(0)
        elif 'boxing' in train_videos[i]:
            train_labels.append(1)
        else:
            train_labels.append(2)

    test_labels = []
    for i in range(len(test_videos)):
        if 'running' in test_videos[i]:
            test_labels.append(0)
        elif 'boxing' in test_videos[i]:
            test_labels.append(1)
        else:
            test_labels.append(2)

    return train_videos, train_labels, test_videos, test_labels


# calculate  HoG/HoF descriptors for every video in dataset and store in local disk
def calc_hog_hof_all(training_videos, test_videos, sigma, s, tau, k, N, nbins):
    for vid in training_videos:
        print(vid)
        video = read_video(vid)
        name = vid.split('/')[3]
        fd1 = open('pickle_data/' + name[:-4] + "_harris_bow.p", 'wb')
        fd2 = open('pickle_data/' + name[:-4] + "_gabor_bow.p", 'wb')

        H1 = harris_3d(video, sigma, s, tau, k)
        H2 = gabor(video, sigma, tau)

        int_pts1 = interest_points_extraction(H1, N, sigma)
        int_pts2 = interest_points_extraction(H2, N, sigma)

        desc1 = get_HoG_HoF(video, int_pts1, sigma, nbins)
        desc2 = get_HoG_HoF(video, int_pts2, sigma, nbins)

        pickle.dump(desc1, fd1)
        pickle.dump(desc2, fd2)
        fd1.close()
        fd2.close()

    for vid in test_videos:
        print(vid)
        video = read_video(vid)
        name = vid.split('/')[3]
        fd1 = open('pickle_data/' + name[:-4] + "_harris_bow.p", 'wb')
        fd2 = open('pickle_data/' + name[:-4] + "_gabor_bow.p", 'wb')

        H1 = harris_3d(video, sigma, s, tau, k)
        H2 = gabor(video, sigma, tau)

        int_pts1 = interest_points_extraction(H1, N, sigma)
        int_pts2 = interest_points_extraction(H2, N, sigma)

        desc1 = get_HoG_HoF(video, int_pts1, sigma, nbins)
        desc2 = get_HoG_HoF(video, int_pts2, sigma, nbins)

        pickle.dump(desc1, fd1)
        pickle.dump(desc2, fd2)
        fd1.close()
        fd2.close()


# calculate  HoG descriptors for every video in dataset and store in local disk
def calc_hog_all(training_videos, test_videos, sigma, s, tau, k, N, nbins):
    for vid in training_videos:
        print(vid)
        video = read_video(vid)
        name = vid.split('/')[3]
        fd1 = open('pickle_data_hog/' + name[:-4] + "_harris_bow.p", 'wb')
        fd2 = open('pickle_data_hog/' + name[:-4] + "_gabor_bow.p", 'wb')

        H1 = harris_3d(video, sigma, s, tau, k)
        H2 = gabor(video, sigma, tau)

        int_pts1 = interest_points_extraction(H1, N, sigma)
        int_pts2 = interest_points_extraction(H2, N, sigma)

        desc1 = get_HoG(video, int_pts1, sigma, nbins)
        desc2 = get_HoG(video, int_pts2, sigma, nbins)

        pickle.dump(desc1, fd1)
        pickle.dump(desc2, fd2)
        fd1.close()
        fd2.close()

    for vid in test_videos:
        print(vid)
        video = read_video(vid)
        name = vid.split('/')[3]
        fd1 = open('pickle_data_hog/' + name[:-4] + "_harris_bow.p", 'wb')
        fd2 = open('pickle_data_hog/' + name[:-4] + "_gabor_bow.p", 'wb')

        H1 = harris_3d(video, sigma, s, tau, k)
        H2 = gabor(video, sigma, tau)

        int_pts1 = interest_points_extraction(H1, N, sigma)
        int_pts2 = interest_points_extraction(H2, N, sigma)

        desc1 = get_HoG(video, int_pts1, sigma, nbins)
        desc2 = get_HoG(video, int_pts2, sigma, nbins)

        pickle.dump(desc1, fd1)
        pickle.dump(desc2, fd2)
        fd1.close()
        fd2.close()


# calculate  HoF descriptors for every video in dataset and store in local disk
def calc_hof_all(training_videos, test_videos, sigma, s, tau, k, N, nbins):
    for vid in training_videos:
        print(vid)
        video = read_video(vid)
        name = vid.split('/')[3]
        fd1 = open('pickle_data_hof/' + name[:-4] + "_harris_bow.p", 'wb')
        fd2 = open('pickle_data_hof/' + name[:-4] + "_gabor_bow.p", 'wb')

        H1 = harris_3d(video, sigma, s, tau, k)
        H2 = gabor(video, sigma, tau)

        int_pts1 = interest_points_extraction(H1, N, sigma)
        int_pts2 = interest_points_extraction(H2, N, sigma)

        desc1 = get_HoF(video, int_pts1, sigma, nbins)
        desc2 = get_HoF(video, int_pts2, sigma, nbins)

        pickle.dump(desc1, fd1)
        pickle.dump(desc2, fd2)
        fd1.close()
        fd2.close()

    for vid in test_videos:
        print(vid)
        video = read_video(vid)
        name = vid.split('/')[3]
        fd1 = open('pickle_data_hof/' + name[:-4] + "_harris_bow.p", 'wb')
        fd2 = open('pickle_data_hof/' + name[:-4] + "_gabor_bow.p", 'wb')

        H1 = harris_3d(video, sigma, s, tau, k)
        H2 = gabor(video, sigma, tau)

        int_pts1 = interest_points_extraction(H1, N, sigma)
        int_pts2 = interest_points_extraction(H2, N, sigma)

        desc1 = get_HoF(video, int_pts1, sigma, nbins)
        desc2 = get_HoF(video, int_pts2, sigma, nbins)

        pickle.dump(desc1, fd1)
        pickle.dump(desc2, fd2)
        fd1.close()
        fd2.close()


# return HoG/HoF, HoG or HoF descriptors (specified in arguments) for every video in dataset from local storage
# must run after calc_hog_hof_all(), calc_hog_all() or calc_hof_all()
def fetch_hog_hof_all(training_videos, test_videos, descriptor=''):
    train1 = []
    train2 = []
    assert descriptor in ['', '_hog', '_hof']
    for vid in training_videos:
        name = vid.split('/')[3]
        fd1 = open('pickle_data' + descriptor + '/' + name[:-4] + "_harris_bow.p", 'rb')
        fd2 = open('pickle_data' + descriptor + '/' + name[:-4] + "_gabor_bow.p", 'rb')

        desc1 = pickle.load(fd1)
        train1.append(desc1)
        desc2 = pickle.load(fd2)
        train2.append(desc2)

        fd1.close()
        fd2.close()

    test1 = []
    test2 = []
    for vid in test_videos:
        name = vid.split('/')[3]
        fd1 = open('pickle_data' + descriptor + '/' + name[:-4] + "_harris_bow.p", 'rb')
        fd2 = open('pickle_data' + descriptor + '/' + name[:-4] + "_gabor_bow.p", 'rb')

        desc1 = pickle.load(fd1)
        test1.append(desc1)
        desc2 = pickle.load(fd2)
        test2.append(desc2)

        fd1.close()
        fd2.close()

    return train1, test1, train2, test2


# for the training/testing partitioning of videos and specified descriptor use SVM
# to predict label of video and measure accuracy
# repeat a number of times to find mean accuracy
def train(tr_videos, tr_labels, tst_videos, tst_labels, iterations, descriptor):
    tr_harris, tst_harris, tr_gabor, tst_gabor = fetch_hog_hof_all(tr_videos, tst_videos, descriptor)

    tot_acc_har = 0
    tot_acc_gab = 0
    print("Correct labels: ", tst_labels)
    for i in range(iterations):
        bow_tr_har, bow_tst_har = bag_of_words(tr_harris, tst_harris, 50)
        bow_tr_gabor, bow_tst_gabor = bag_of_words(tr_gabor, tst_gabor, 50)
        acc_har, pred_har = svm_train_test(bow_tr_har, tr_labels, bow_tst_har, tst_labels)
        print("Prediction ", i+1, " for harris:\t", pred_har)
        acc_gab, pred_gab = svm_train_test(bow_tr_gabor, tr_labels, bow_tst_gabor, tst_labels)
        print("Prediction ", i+1, " for gabor:\t", pred_gab)
        tot_acc_har += acc_har
        tot_acc_gab += acc_gab
    tot_acc_har /= iterations
    tot_acc_gab /= iterations
    print("Total accuracy for Harris detector: ", tot_acc_har * 100, "%")
    print("Total accuracy for Gabor detector: ", tot_acc_gab * 100, "%")


tr_vid, tr_lbl, test_vid, test_lbl = train_test()
# calc_hog_all(tr_vid, test_vid, 4, 2, 1.5, 0.005, 500, 10)
# calc_hof_all(tr_vid, test_vid, 4, 2, 1.5, 0.005, 500, 10)
# calc_hog_hof_all(tr_vid, test_vid, 4, 2, 1.5, 0.005, 1200, 10)
train(tr_vid, tr_lbl, test_vid, test_lbl, 10, '_hof')
