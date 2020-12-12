import imutils
import numpy as np
from imutils import contours
import cv2


# input question_no to return first index of choice each of questions
def choice_to_bubble(choice, amount, col):
    final_question_col = int(amount / col)
    column = (4 if choice > final_question_col * 4 else
              3 if choice > final_question_col * 3 else
              2 if choice > final_question_col * 2 else
              1 if choice > final_question_col else
              0)
    return (choice - (final_question_col * column + 1)) * (5*col) + (5 * column + 1)


# check to index of bubble choice each questions
def calulate_score(questions_no, answer_sheet, subject_img, amount, col, dic_c_form):
    first_bubble = choice_to_bubble(questions_no, amount, col)
    answer_choice = list(filter(lambda bubbles: first_bubble + 4 >= bubbles[1] >= first_bubble, answer_sheet))
    form_choice = list(filter(lambda bubbles: first_bubble + 4 >= bubbles[1] >= first_bubble, dic_c_form))
    chosen_choice = None
    chosen_pos = 0
    total_c_form = 0
    count_bubble = 0
    for i, c_form in enumerate(form_choice):
        total_c_form += c_form[0]
    avg = total_c_form / 5
    for pos, bubble in enumerate(answer_choice):
        diff = (abs(avg - bubble[0]))
        if not chosen_choice and diff < 40:
            chosen_choice = bubble
            chosen_pos = pos
            count_bubble += 1
            continue
        if not chosen_choice and diff >= 40:
            chosen_choice = bubble
            chosen_pos = -1
            continue
        if bubble[0] >= chosen_choice[0] and diff < 40:
            chosen_choice = bubble
            chosen_pos = pos
            count_bubble += 1
        if bubble[0] < chosen_choice[0] and diff < 40:
            count_bubble += 1
    solve_pos = draw_answer(subject_img, chosen_pos + 1, answer_choice, count_bubble)
    return {
        'position_solve': solve_pos
    }


# draw choices with correct answer each questions
def draw_answer(img, pos, list_answer_choice, count):
    solve_pos = 0
    if pos > 0 and count == 1:
        solve_pos = pos
        cv2.drawContours(img, [list_answer_choice[pos - 1][2]], -1, (0, 255, 0), 3)
    return solve_pos


def detect_circle(img_gray, num_choices):
    thresh = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 71, 2 | cv2.THRESH_OTSU)
    # หา contour ที่เป็น choices คำตอบทั้งหมด
    contour = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contour = imutils.grab_contours(contour)
    contour = sorted(contour, key=cv2.contourArea, reverse=True)
    questionCnts = []
    # loop over the contours
    for c in contour:
        # compute the bounding box of the contour, then use the
        # bounding box to derive the aspect ratio
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.03 * peri, True)
        (x, y, w, h) = cv2.boundingRect(c)
        # in order to label the contour as a question, region
        # should be sufficiently wide, sufficiently tall, and
        # have an aspect ratio approximately equal to 1
        if 15 <= w <= 50 and 15 <= h <= 50 \
                and len(approx) != 3 and len(approx) != 4 and len(approx) != 5:
            questionCnts.append(c)
    print(len(questionCnts))
    if len(questionCnts) > num_choices:
        skip = len(questionCnts) - num_choices
        questionCnts = questionCnts[:-skip]
    print(len(questionCnts))
    return questionCnts


def subtract_img(list_question_cnts, subject_img, form_img):
    blanked_img = np.zeros(form_img.shape, dtype="uint8")
    new_marker = cv2.drawContours(blanked_img, list_question_cnts, -1, (255, 255, 255), -1)
    new_marker_gray = cv2.cvtColor(new_marker, cv2.COLOR_BGR2GRAY)
    # cv2.imwrite('Choices/Case6/2marker.jpg', new_marker_gray)
    # cv2.imshow('new_marker_gray', new_marker_gray)
    subject_gray = cv2.cvtColor(subject_img, cv2.COLOR_BGR2GRAY)
    subject_gray_blurred = cv2.GaussianBlur(subject_gray, (5, 5), 0)
    th1 = cv2.adaptiveThreshold(subject_gray_blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 71,
                                2 | cv2.THRESH_OTSU)
    # cv2.imshow('thres_subject', th1)
    # cv2.imwrite('Choices/Case6/3thres_subject.jpg', th1)
    new_sub = cv2.bitwise_and(new_marker_gray, th1)
    # cv2.imshow('new_sub', new_sub)
    # cv2.imwrite('Choices/Case6/4new_sub.jpg', new_sub)
    return {
        'new_sub': new_sub,
        'new_marker_gray': new_marker_gray
    }


def find_circle_contour(bound_img, num_choices):
    contour = cv2.findContours(bound_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = imutils.grab_contours(contour)
    contour = sorted(contour, key=cv2.contourArea, reverse=True)
    circleCnts = []
    # loop over the contours
    for c in contour:
        # compute the bounding box of the contour, then use the
        # bounding box to derive the aspect ratio
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.03 * peri, True)
        (x, y, w, h) = cv2.boundingRect(c)
        ar = w / float(h)
        # in order to label the contour as a question, region
        # should be sufficiently wide, sufficiently tall, and
        # have an aspect ratio approximately equal to 1
        if 15 <= w <= 50 and 15 <= h <= 50 \
                and len(approx) != 3 and len(approx) != 4 and len(approx) != 5:
            circleCnts.append(c)
    print(len(circleCnts))
    if len(circleCnts) > num_choices:
        skip = len(circleCnts) - num_choices
        circleCnts = circleCnts[:-skip]
    print(len(circleCnts))
    return circleCnts


def mask_choices_bubbled(choice_contours, bound_img, col):
    questnts = contours.sort_contours(choice_contours, method="top-to-bottom")[0]
    list_bubbled = []
    # loop แต่ละแถว ซึ่ง 1 แถวมี 20 choices (5ข้อ) วนไปจนครบ 25 แถว
    # เพื่อ sort contours จากซ้ายไปขวา
    for (q, i) in enumerate(np.arange(0, len(questnts), 5*col)):
        # sort the contours for the current question from
        # left to right, then initialize the index of the
        # bubbled answer
        cnts = contours.sort_contours(questnts[i:i + 5*col])[0]
        bubbled = None
        # for แต่ละ contours เพื่อ mask และนำค่า mask แต่ละ choices
        # เข้าใน list_bubbled
        for (j, c) in enumerate(cnts):
            # construct a mask that reveals only the current
            # "bubble" for the question
            mask = np.zeros(bound_img.shape, dtype="uint8")
            ex1 = cv2.drawContours(mask, [c], -1, 255, -1)
            # apply the mask to the thresholded image, then
            # count the number of non-zero pixels in the
            # bubble area
            mask = cv2.bitwise_or(bound_img, bound_img, mask=mask)
            total = cv2.countNonZero(mask)
            # if the current total has a larger number of total
            # non-zero pixels, then we are examining the currently
            # bubbled-in answer
            bubbled = (total, i + j + 1, c)
            list_bubbled.append(bubbled)
    return list_bubbled


def main_process(form_img, subject_img, quiz, amount_choices, column):
    image = cv2.imread(form_img)
    subject = subject_img
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    subject_gray = cv2.cvtColor(subject, cv2.COLOR_BGR2GRAY)
    check_ans_cnts = detect_circle(subject_gray, 1000)
    if len(check_ans_cnts) >= amount_choices * 5:
        questionCnts = detect_circle(gray, amount_choices*5)
        new_subject_image = subtract_img(questionCnts, subject, image)
        boundImg = cv2.drawContours(new_subject_image['new_sub'].copy(), questionCnts, -1, (255, 255, 255), 1)
        choicesCnts = find_circle_contour(boundImg, amount_choices*5)
        list_choices_bubbled = mask_choices_bubbled(choicesCnts, boundImg, column)
        dict_c_form = mask_choices_bubbled(questionCnts, new_subject_image['new_marker_gray'], column)

        # loop for calculate score
        dict_result = {}
        for i in range(1, quiz['amount']+1):
            result = calulate_score(i, list_choices_bubbled, subject, amount_choices, column, dict_c_form)
            if result['position_solve'] > 0:
                dict_result[str(i)] = result['position_solve']
        return {
            'result_solve': dict_result,
            'img_solve': subject
        }
    else:
        return {
            'error_msg': 'ไม่สามารถตรวจข้อสอบได้เพราะ align รูปส่วนฝนคำตอบได้ไม่ถูกต้อง',
            'is_error': True
        }

