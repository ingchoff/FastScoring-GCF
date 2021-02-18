import cv2
import numpy as np
from exam import Compare
from exam import MsePaper
from exam import ImgProcess

GOOD_MATCH_PERCENT = 0.15


def alignImages(im1, im2, type_sift, plus, descriptors1, descriptors2, keypoints1, keypoints2):
    # Match features.
    # bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    # matches = bf.match(descriptors1, descriptors2)
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors1, descriptors2, None)
    # bf = cv2.BFMatcher()
    # matches = bf.knnMatch(descriptors1, descriptors2, k=2)

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


def find_compare(rounds, list_mse_prv, img, img_gray, img_refer, img_refer_gray):
    is_loop = True
    increase = 0
    list_mse = []
    start_feature = 0
    end_feature = 0
    obj_aligned = {}
    if rounds == 1:
        start_feature = 1000
        end_feature = 2500
    elif rounds == 2:
        start_feature = 2600
        end_feature = 5000
        list_mse = list_mse_prv
    elif rounds == 3:
        start_feature = 1000
        end_feature = 5000
        list_mse = list_mse_prv
    while is_loop:
        # Registered image will be resotred in imReg.
        # The estimated homography will be stored in h.
        print(start_feature + increase)
        orb = cv2.ORB_create(start_feature + increase)
        keypoints1, descriptors1 = orb.detectAndCompute(img_gray, None)
        keypoints2, descriptors2 = orb.detectAndCompute(img_refer_gray, None)
        result_aligned = alignImages(img, img_refer, 'std', increase, descriptors1, descriptors2, keypoints1,
                                     keypoints2)
        if result_aligned['is_error']:
            print(result_aligned['is_error'])
        else:
            is_aligned_pass = Compare.main_process(result_aligned['aligned_img'], img_refer, start_feature+increase, rounds)
            if is_aligned_pass["is_aligned"]:
                obj_aligned["aligned_img"] = result_aligned['aligned_img']
                obj_aligned["list_mse"] = list_mse
                is_loop = False
            else:
                list_mse.append(is_aligned_pass["aligned_value"])
                if (start_feature + increase) == end_feature:
                    is_loop = False
                    increase = 0
                increase += 100
    obj_aligned["list_mse"] = list_mse
    print(obj_aligned["list_mse"])
    return obj_aligned


def main_process(img_form, img_subject, answer_coords, stu_coords, type_align, option_round):
    imReference = cv2.imread(img_form, cv2.IMREAD_COLOR)
    # Read image to be aligned
    im = cv2.imread(img_subject, cv2.IMREAD_COLOR)
    increase = 0
    # Convert images to grayscale
    im1Gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im2Gray = cv2.cvtColor(imReference, cv2.COLOR_BGR2GRAY)
    if option_round == 1:
        obj_aligned_final = find_compare(3, [], im, im1Gray, imReference, im2Gray)
        x_ans = int(answer_coords['x'])
        y_ans = int(answer_coords['y'])
        w_ans = int(answer_coords['width'])
        h_ans = int(answer_coords['height'])
        x_stu = int(stu_coords['x'])
        y_stu = int(stu_coords['y'])
        w_stu = int(stu_coords['width'])
        h_stu = int(stu_coords['height'])
        if "aligned_img" in obj_aligned_final:
            answer_crop_img = obj_aligned_final['aligned_img'][y_ans:y_ans + h_ans, x_ans:x_ans + w_ans]
            stu_crop_img = obj_aligned_final['aligned_img'][y_stu:y_stu + h_stu, x_stu:x_stu + w_stu]
            return {
                'answer_aligned_img': answer_crop_img,
                'stu_aligned_img': stu_crop_img,
                'is_error': False
            }
        else:
            obj_aligned = find_compare(1, [], im, im1Gray, imReference, im2Gray)
            result_selected = MsePaper.main(obj_aligned["list_mse"])
            print("feature selected:" + str(result_selected["feature"]))
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
                answer_crop_img_gray = cv2.cvtColor(answer_crop_img, cv2.COLOR_BGR2GRAY)
                stu_crop_img = result_aligned['aligned_img'][y_stu:y_stu + h_stu, x_stu:x_stu + w_stu]
                check_circle = ImgProcess.detect_circle(answer_crop_img_gray, 500, 'exam')
                if len(check_circle) >= 500:
                    return {
                        'answer_aligned_img': answer_crop_img,
                        'stu_aligned_img': stu_crop_img,
                        'is_error': False
                    }
                else:
                    obj_aligned_final = find_compare(2, obj_aligned["list_mse"], im, im1Gray, imReference, im2Gray)
                    result_selected = MsePaper.main(obj_aligned_final["list_mse"])
                    print("feature selected:" + str(result_selected["feature"]))
                    orb = cv2.ORB_create(result_selected["feature"])
                    keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
                    keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)
                    result_aligned = alignImages(im, imReference, 'std', increase, descriptors1, descriptors2,
                                                 keypoints1, keypoints2)
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
    elif option_round == 2:
        obj_aligned = find_compare(1, [], im, im1Gray, imReference, im2Gray)
        result_selected = MsePaper.main(obj_aligned["list_mse"])
        print("feature selected:" + str(result_selected["feature"]))
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
            answer_crop_img_gray = cv2.cvtColor(answer_crop_img, cv2.COLOR_BGR2GRAY)
            stu_crop_img = result_aligned['aligned_img'][y_stu:y_stu + h_stu, x_stu:x_stu + w_stu]
            check_circle = ImgProcess.detect_circle(answer_crop_img_gray, 500, 'exam')
            if len(check_circle) >= 500:
                return {
                    'answer_aligned_img': answer_crop_img,
                    'stu_aligned_img': stu_crop_img,
                    'is_error': False
                }
            else:
                obj_aligned_final = find_compare(2, obj_aligned["list_mse"], im, im1Gray, imReference, im2Gray)
                result_selected = MsePaper.main(obj_aligned_final["list_mse"])
                print("feature selected:" + str(result_selected["feature"]))
                orb = cv2.ORB_create(result_selected["feature"])
                keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
                keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)
                result_aligned = alignImages(im, imReference, 'std', increase, descriptors1, descriptors2,
                                             keypoints1, keypoints2)
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

    # obj_aligned = find_compare(1, [], im, im1Gray, imReference, im2Gray)
    # result_selected = MsePaper.main(obj_aligned["list_mse"])
    # print("feature selected:" + str(result_selected["feature"]))
    # orb = cv2.ORB_create(result_selected["feature"])
    # keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
    # keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)
    # result_aligned = alignImages(im, imReference, 'std', increase, descriptors1, descriptors2, keypoints1,
    #                              keypoints2)
    # if result_aligned['is_error']:
    #     return {
    #         'error_msg': result_aligned['error_msg'],
    #         'is_error': True
    #     }
    # else:
    #     x_ans = int(answer_coords['x'])
    #     y_ans = int(answer_coords['y'])
    #     w_ans = int(answer_coords['width'])
    #     h_ans = int(answer_coords['height'])
    #     x_stu = int(stu_coords['x'])
    #     y_stu = int(stu_coords['y'])
    #     w_stu = int(stu_coords['width'])
    #     h_stu = int(stu_coords['height'])
    #     answer_crop_img = result_aligned['aligned_img'][y_ans:y_ans + h_ans, x_ans:x_ans + w_ans]
    #     answer_crop_img_gray = cv2.cvtColor(answer_crop_img, cv2.COLOR_BGR2GRAY)
    #     stu_crop_img = result_aligned['aligned_img'][y_stu:y_stu + h_stu, x_stu:x_stu + w_stu]
    #     check_circle = ImgProcess.detect_circle(answer_crop_img_gray, 500, 'exam')
    #     if len(check_circle) >= 500:
    #         return {
    #             'answer_aligned_img': answer_crop_img,
    #             'stu_aligned_img': stu_crop_img,
    #             'is_error': False
    #         }
    #     else:
    #         # obj_aligned_final = find_compare(1, [], im, im1Gray, imReference, im2Gray)
    #         obj_aligned = find_compare(2, obj_aligned["list_mse"], im, im1Gray, imReference, im2Gray)
    #         # x_ans = int(answer_coords['x'])
    #         # y_ans = int(answer_coords['y'])
    #         # w_ans = int(answer_coords['width'])
    #         # h_ans = int(answer_coords['height'])
    #         # x_stu = int(stu_coords['x'])
    #         # y_stu = int(stu_coords['y'])
    #         # w_stu = int(stu_coords['width'])
    #         # h_stu = int(stu_coords['height'])
    #         # answer_crop_img = obj_aligned_final['aligned_img'][y_ans:y_ans + h_ans, x_ans:x_ans + w_ans]
    #         # stu_crop_img = obj_aligned_final['aligned_img'][y_stu:y_stu + h_stu, x_stu:x_stu + w_stu]
    #         # return {
    #         #     'answer_aligned_img': answer_crop_img,
    #         #     'stu_aligned_img': stu_crop_img,
    #         #     'is_error': False
    #         # }
    #         result_selected = MsePaper.main(obj_aligned["list_mse"])
    #         print("feature selected:" + str(result_selected["feature"]))
    #         orb = cv2.ORB_create(result_selected["feature"])
    #         keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
    #         keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)
    #         result_aligned = alignImages(im, imReference, 'std', increase, descriptors1, descriptors2,
    #                                      keypoints1,
    #                                      keypoints2)
    #         if result_aligned['is_error']:
    #             return {
    #                 'error_msg': result_aligned['error_msg'],
    #                 'is_error': True
    #             }
    #         else:
    #             x_ans = int(answer_coords['x'])
    #             y_ans = int(answer_coords['y'])
    #             w_ans = int(answer_coords['width'])
    #             h_ans = int(answer_coords['height'])
    #             x_stu = int(stu_coords['x'])
    #             y_stu = int(stu_coords['y'])
    #             w_stu = int(stu_coords['width'])
    #             h_stu = int(stu_coords['height'])
    #             answer_crop_img = result_aligned['aligned_img'][y_ans:y_ans + h_ans, x_ans:x_ans + w_ans]
    #             answer_crop_img_gray = cv2.cvtColor(answer_crop_img, cv2.COLOR_BGR2GRAY)
    #             stu_crop_img = result_aligned['aligned_img'][y_stu:y_stu + h_stu, x_stu:x_stu + w_stu]
    #             check_circle = ImgProcess.detect_circle(answer_crop_img_gray, 500, 'exam')
    #             if len(check_circle) >= 500:
    #                 return {
    #                     'answer_aligned_img': answer_crop_img,
    #                     'stu_aligned_img': stu_crop_img,
    #                     'is_error': False
    #                 }
    #             else:
    #                 obj_aligned_final = find_compare(3, obj_aligned["list_mse"], im, im1Gray, imReference, im2Gray)
    #                 x_ans = int(answer_coords['x'])
    #                 y_ans = int(answer_coords['y'])
    #                 w_ans = int(answer_coords['width'])
    #                 h_ans = int(answer_coords['height'])
    #                 x_stu = int(stu_coords['x'])
    #                 y_stu = int(stu_coords['y'])
    #                 w_stu = int(stu_coords['width'])
    #                 h_stu = int(stu_coords['height'])
    #                 answer_crop_img = obj_aligned_final['aligned_img'][y_ans:y_ans + h_ans, x_ans:x_ans + w_ans]
    #                 stu_crop_img = obj_aligned_final['aligned_img'][y_stu:y_stu + h_stu, x_stu:x_stu + w_stu]
    #                 return {
    #                     'answer_aligned_img': answer_crop_img,
    #                     'stu_aligned_img': stu_crop_img,
    #                     'is_error': False
    #                 }
