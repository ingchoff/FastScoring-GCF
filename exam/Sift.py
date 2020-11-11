import cv2
import numpy as np

GOOD_MATCH_PERCENT = 0.15


def alignImages(im1, im2):
    # Convert images to grayscale
    im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2Gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    # Detect Sift features and compute descriptors.
    sift = cv2.SIFT_create()
    keypoints1, descriptors1 = sift.detectAndCompute(im1Gray, None)
    keypoints2, descriptors2 = sift.detectAndCompute(im2Gray, None)

    # Match features.
    # matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    # matches = matcher.match(descriptors1, descriptors2, None)
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)

    # Sort matches by score
    # matches.sort(key=lambda x: x.distance, reverse=False)

    # Remove not so good matches
    # numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
    # matches = matches[:numGoodMatches]
    good_matches = []
    for m, n in matches:
        if m.distance < 0.5*n.distance:
            good_matches.append([m])

    # Draw top matches
    # imMatches = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)
    # imMatches = cv2.drawMatchesKnn(im1, keypoints1, im2, keypoints2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    # cv2.imwrite("Aligned/case1-match-sift.jpg", imMatches)

    # Extract location of good matches
    # points1 = np.zeros((len(matches), 2), dtype=np.float32)
    # points2 = np.zeros((len(matches), 2), dtype=np.float32)
    points1 = np.float32([keypoints1[m[0].queryIdx].pt for m in good_matches])
    points2 = np.float32([keypoints2[m[0].trainIdx].pt for m in good_matches])

    # Find homography
    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

    # Use homography
    height, width, channels = im2.shape
    im1Reg = cv2.warpPerspective(im1, h, (width, height))

    return im1Reg, h


def main_process(img_form, img_subject):
    imReference = cv2.imread(img_form, cv2.IMREAD_COLOR)
    # cv2.imshow('test', imReference)
    # Read image to be aligned
    im = cv2.imread(img_subject, cv2.IMREAD_COLOR)

    print("Aligning images ...")
    # Registered image will be resotred in imReg.
    # The estimated homography will be stored in h.
    imReg, h = alignImages(im, imReference)

    # Write aligned image to disk.
    # print("Saving aligned image : ", args["destination"])
    # cv2.imwrite(args["destination"], imReg)

    # Print estimated homography
    print("Estimated homography : \n", h)
    return imReg
