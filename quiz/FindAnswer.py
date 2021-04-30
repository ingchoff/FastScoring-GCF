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
def calulate_score(questions_no, answersheet, subject_img, amount, col, dic_c_form):
    first_bubble = choice_to_bubble(questions_no, amount, col)
    answer_choice = list(filter(lambda bubbles: first_bubble + 4 >= bubbles[1] >= first_bubble, answersheet))
    form_choice = list(filter(lambda bubbles: first_bubble + 4 >= bubbles[1] >= first_bubble, dic_c_form))
    chosen_choice = None
    chosen_pos = 0
    total_c_form = 0
    list_diff = []
    list_selected = []
    list_bubble = []
    for i, c_form in enumerate(form_choice):
        total_c_form += c_form[0]
    avg = int(total_c_form / 5)
    for bubble in answer_choice:
        diff = (abs(avg - bubble[0]))
        list_diff.append(diff)
        list_bubble.append(bubble[0])
    for pos, bubble in enumerate(answer_choice):
        if not chosen_choice:
            chosen_choice = bubble
            chosen_pos = pos
            continue
        if bubble[0] >= chosen_choice[0]:
            chosen_choice = bubble
            chosen_pos = pos
    list_selected.append(chosen_pos)
    print(list_bubble)
    print(list_diff)
    print(list_selected)
    for pos, diff in enumerate(list_diff):
        if diff <= 60 and pos not in list_selected:
            list_selected.append(pos)
        elif diff > 60 and pos in list_selected:
            percent = (list_bubble[pos] / avg) * 100
            print(percent)
            if percent <= 45:
                list_selected.remove(pos)
            if percent > 45 and pos != 4 and abs(list_diff[pos]-list_diff[pos-1]) <= 42 and abs(list_diff[pos]-list_diff[pos+1]) <= 42:
                list_selected.remove(pos)
            if percent > 45 and pos == 4 and abs(list_diff[pos]-list_diff[pos-1]) <= 42 and abs(list_diff[pos]-list_diff[0]) <= 42:
                list_selected.remove(pos)
    print(list_selected)
    solve_pos = draw_answer(subject_img, sorted(list_selected), answer_choice)
    return {
        'position_solve': solve_pos
    }


# draw choices with correct answer each questions
def draw_answer(img, list_pos, list_answer_choice):
    # solve_pos = -1
    solve_pos = []
    for count, pos in enumerate(list_pos):
        solve_pos.append(pos + 1)
        cv2.drawContours(img, [list_answer_choice[pos][2]], -1, (0, 255, 0), 2)
    # if len(list_pos) == 1:
    #     # solve_pos = list_pos[0]
    #     solve_pos.append(list_pos[0] + 1)
    #     cv2.drawContours(img, [list_answer_choice[solve_pos][2]], -1, (0, 255, 0), 2)
    # elif len(list_pos) > 1:
    #     for count, pos in enumerate(list_pos):
    #         # solve_pos = pos
    #         solve_pos.append(pos + 1)
    #         cv2.drawContours(img, [list_answer_choice[solve_pos][2]], -1, (0, 255, 0), 2)
    # return solve_pos + 1
    return solve_pos

def detect_circle(img_gray, num_choices):
    thresh = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 71, 2 | cv2.THRESH_OTSU)
    # cv2.imshow('th', thresh)
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


def subtract_img(list_question_cnts, subject_gray, form_img):
    blanked_img = np.zeros(form_img.shape, dtype="uint8")
    new_marker = cv2.drawContours(blanked_img, list_question_cnts, -1, (255, 255, 255), -1)
    new_marker_gray = cv2.cvtColor(new_marker, cv2.COLOR_BGR2GRAY)
    subject_gray_blurred = cv2.GaussianBlur(subject_gray, (5, 5), 0)
    th1 = cv2.adaptiveThreshold(subject_gray_blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 71,
                                2 | cv2.THRESH_OTSU)
    new_sub = cv2.bitwise_and(new_marker_gray, th1)
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


def main_process(form_img, subject_img, quiz, amount_choices, column, coords, num_choice):
    form = cv2.imread(form_img)
    x = int(coords['x'])
    y = int(coords['y'])
    w = int(coords['width'])
    h = int(coords['height'])
    crop_form_img = form[y:y + h, x:x + w]
    subject = subject_img
    gray = cv2.cvtColor(crop_form_img, cv2.COLOR_BGR2GRAY)
    subject_gray = cv2.cvtColor(subject, cv2.COLOR_BGR2GRAY)
    check_ans_cnts = detect_circle(subject_gray, 1000)
    if len(check_ans_cnts) >= amount_choices * num_choice:
        questionCnts = detect_circle(gray, amount_choices*num_choice)
        new_subject_image = subtract_img(questionCnts, subject_gray, crop_form_img)
        boundImg = cv2.drawContours(new_subject_image['new_sub'].copy(), questionCnts, -1, (255, 255, 255), 1)
        choicesCnts = find_circle_contour(boundImg, amount_choices*num_choice)
        list_choices_bubbled = mask_choices_bubbled(choicesCnts, boundImg, column)
        dict_c_form = mask_choices_bubbled(questionCnts, new_subject_image['new_marker_gray'], column)

        # loop for calculate score
        dict_result = {}
        for i in range(1, quiz['amount']+1):
            result = calulate_score(i, list_choices_bubbled, subject, amount_choices, column, dict_c_form)
            if len(result['position_solve']) > 0:
                dict_result[str(i)] = result['position_solve']
        return {
            'is_error': False,
            'result_solve': dict_result,
            'img_solve': subject
        }
    else:
        return {
            'error_msg': 'ไม่สามารถตรวจหาเฉลยได้เพราะ align รูปส่วนฝนคำตอบได้ไม่ถูกต้อง',
            'is_error': True
        }

