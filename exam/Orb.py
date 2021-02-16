import cv2
import numpy as np
from exam import Compare
from exam import MsePaper

GOOD_MATCH_PERCENT = 0.15


def alignImages(im1, im2, type_sift, plus, descriptors1, descriptors2, keypoints1, keypoints2):
    # Match features.
    # bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    # matches = bf.match(descriptors1, descriptors2)
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors1, descriptors2, None)
    # bf = cv2.BFMatcher()
    # matches = bf.knnMatch(descriptors1, descriptors2, k=2)
    print(len(matches))

    # Sort matches by score
    matches = sorted(matches, key=lambda x: x.distance)
    # plus_next = (plus * 14)

    # Remove not so good matches
    numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
    # matches = matches[0 + plus_next:13 + plus_next]
    matches = matches[:numGoodMatches]
    # good_matches = []
    # for m, n in matches:
    #     if m.distance < 0.5 * n.distance:
    #         good_matches.append([m])
    # print(len(good_matches))

    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i] = keypoints1[match.queryIdx].pt
        points2[i] = keypoints2[match.trainIdx].pt

    # points1 = np.float32([keypoints1[m[0].queryIdx].pt for m in good_matches])
    # points2 = np.float32([keypoints2[m[0].trainIdx].pt for m in good_matches])
    try:
        # Find homography
        h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

        # Use homography
        height, width, channels = im2.shape
        im1Reg = cv2.warpPerspective(im1, h, (width, height))
        return {
            'is_error': False,
            'aligned_img': im1Reg,
            'h': h
        }
    except cv2.error:
        return {
            'is_error': True,
            'error_msg': cv2.error.msg
        }


def main_process(img_form, img_subject, answer_coords, stu_coords, type_align):
    imReference = cv2.imread(img_form, cv2.IMREAD_COLOR)
    # Read image to be aligned
    im = cv2.imread(img_subject, cv2.IMREAD_COLOR)
    is_loop = True
    increase = 0
    # Convert images to grayscale
    im1Gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im2Gray = cv2.cvtColor(imReference, cv2.COLOR_BGR2GRAY)
    # Detect Sift features and compute descriptors.
    # print(500 + plus)
    # orb = cv2.ORB_create(3000)
    # keypoints1, descriptors1 = orb.detectAndCompute(im1blurred, None)
    # keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)
    list_values = []
    while is_loop:
        print("Aligning images ...")
        # print(increase)
        # Registered image will be resotred in imReg.
        # The estimated homography will be stored in h.
        orb = cv2.ORB_create(1000 + increase)
        keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
        keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)
        result_aligned = alignImages(im, imReference, 'std', increase, descriptors1, descriptors2, keypoints1,
                                     keypoints2)
        if result_aligned['is_error']:
            return {
                'error_msg': result_aligned['error_msg'],
                'is_error': True
            }
        else:
            is_aligned_pass = Compare.main_process(result_aligned['aligned_img'], imReference, increase+1000)
            if is_aligned_pass["is_aligned"]:
                # Print estimated homography
                print("Estimated homography : \n", result_aligned['h'])
                x_ans = int(answer_coords['x'])
                y_ans = int(answer_coords['y'])
                w_ans = int(answer_coords['width'])
                h_ans = int(answer_coords['height'])
                x_stu = int(stu_coords['x'])
                y_stu = int(stu_coords['y'])
                w_stu = int(stu_coords['width'])
                h_stu = int(stu_coords['height'])
                answer_crop_img = result_aligned['aligned_img'][y_ans:y_ans + h_ans, x_ans:x_ans + w_ans]
                stu_crop_img = result_aligned['aligned_img'][y_stu:y_stu + h_stu, x_stu:x_stu + w_stu]
                # crop_gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
                # test_grader.detect_circle(crop_gray, 500, 'answer')
                is_loop = False
                return {
                    'answer_aligned_img': answer_crop_img,
                    'stu_aligned_img': stu_crop_img,
                    'is_error': False
                }
            else:
                if (1000 + increase) == 5000:
                    is_loop = False
                list_values.append(is_aligned_pass["aligned_value"])
                increase += 100
                print("re-aligned")
    result_selected = MsePaper.main(list_values)
    print(result_selected["feature"])
    orb = cv2.ORB_create(result_selected["feature"])
    keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)
    result_aligned = alignImages(im, imReference, 'std', increase, descriptors1, descriptors2, keypoints1,
                                 keypoints2)
    if result_aligned['is_error']:
        return {
            'error_msg': result_aligned['error_msg'],
            'is_error': True
        }
    else:
        x_ans = int(answer_coords['x'])
        y_ans = int(answer_coords['y'])
        w_ans = int(answer_coords['width'])
        h_ans = int(answer_coords['height'])
        x_stu = int(stu_coords['x'])
        y_stu = int(stu_coords['y'])
        w_stu = int(stu_coords['width'])
        h_stu = int(stu_coords['height'])
        answer_crop_img = result_aligned['aligned_img'][y_ans:y_ans + h_ans, x_ans:x_ans + w_ans]
        stu_crop_img = result_aligned['aligned_img'][y_stu:y_stu + h_stu, x_stu:x_stu + w_stu]
        return {
            'answer_aligned_img': answer_crop_img,
            'stu_aligned_img': stu_crop_img,
            'is_error': False
        }
