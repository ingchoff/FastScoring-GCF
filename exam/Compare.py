# import the necessary packages
from skimage.metrics import structural_similarity as ssim
import numpy as np
import cv2
from exam import ImgProcess


def mse(imageA, imageB):
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])

    # return the MSE, the lower the error, the more "similar"
    # the two images are
    return err


def compare_images(imageA, imageB, feature, rounds, path):
    # compute the mean squared error and structural similarity
    # index for the images
    is_pass = False
    camera = cv2.adaptiveThreshold(imageB, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 71, 2 | cv2.THRESH_OTSU)
    kernel = np.ones((1, 1), np.uint8)
    camera = cv2.erode(camera, kernel, iterations=1)
    # camera = cv2.bitwise_not(camera)
    cv2.imwrite(path + '/' + 'imageA1.jpg', imageA)
    imageA = cv2.adaptiveThreshold(imageA, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 71, 2 | cv2.THRESH_OTSU)
    cv2.imwrite(path + '/' + 'imageA2.jpg', imageA)
    cv2.imwrite(path + '/' + 'camera2.jpg', camera)
    diff = cv2.bitwise_and(imageA, camera)
    # diff = cv2.bitwise_not(diff)
    m = mse(imageA, diff)
    s = ssim(imageA, diff)
    values_compare = {}
    print("MSE: %.2f, SSIM: %.2f" % (m, s))
    if rounds == 1:
        values_compare["MSE"] = m
        values_compare["SSIM"] = s
        values_compare["feature"] = feature
        cv2.imwrite(path + '/' + 'diff.jpg', diff)
    if rounds == 2:
        values_compare["MSE"] = m
        values_compare["SSIM"] = s
        values_compare["feature"] = feature
        cv2.imwrite(path + '/' + 'diff.jpg', diff)
    # is_pass = True
    elif rounds == 3:
        check_circle = ImgProcess.detect_circle(imageB, 500, 'exam')
        values_compare["MSE"] = m
        values_compare["SSIM"] = s
        values_compare["feature"] = feature
        if s >= 0.8 and len(check_circle) >= 500:
            cv2.imwrite(path + '/' + 'diff.jpg', diff)
            is_pass = True
        else:
            cv2.imwrite(path + '/' + 'diff.jpg', diff)
            is_pass = False
    print(values_compare)
    return {
        "aligned_value": values_compare,
        "is_aligned": is_pass
    }


def main_process(aligned_img, form_img, max_feature, rounds, path):
    # load the images -- the original, the original + contrast,
    # and the original + photoshop
    original = form_img
    camera = aligned_img
    # convert the images to grayscale
    original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    camera = cv2.cvtColor(camera, cv2.COLOR_BGR2GRAY)
    camera = cv2.medianBlur(camera, 1)
    # cv2.imshow('camera', camera)
    # compare two images
    is_aligned = compare_images(original, camera, max_feature, rounds, path)
    return is_aligned
