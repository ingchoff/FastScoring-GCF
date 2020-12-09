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
def calulate_score(questions_no, answersheet, subject_img, answer_keys, amount, col, dic_c_form):
    first_bubble = choice_to_bubble(questions_no, amount, col)
    answer_choice = list(filter(lambda bubbles: first_bubble + 4 >= bubbles[1] >= first_bubble, answersheet))
    form_choice = list(filter(lambda bubbles: first_bubble + 4 >= bubbles[1] >= first_bubble, dic_c_form))
    chosen_choice = None
    chosen_pos = 0
    total_c_form = 0
    diff = []
    count_bubble = 0
    for i, c_form in enumerate(form_choice):
        total_c_form += c_form[0]
    avg = total_c_form / 5
    for pos, bubble in enumerate(answer_choice):
        diff.append(abs(avg - bubble[0]))
        if not chosen_choice and abs(avg - bubble[0]) < 40:
            chosen_choice = bubble
            chosen_pos = pos
            count_bubble += 1
            continue
        if not chosen_choice and abs(avg - bubble[0]) >= 40:
            chosen_choice = bubble
            chosen_pos = -1
            continue
        if bubble[0] >= chosen_choice[0] and abs(avg - bubble[0]) < 40:
            chosen_choice = bubble
            chosen_pos = pos
            count_bubble += 1
        if bubble[0] < chosen_choice[0] and abs(avg - bubble[0]) < 40:
            count_bubble += 1
    correct = draw_answer(questions_no, subject_img, answer_keys, chosen_pos + 1, answer_choice, diff[chosen_pos], count_bubble)
    return {
        'user_choice': chosen_pos + 1,
        'correct': correct['is_correct'],
        'correct_choice': correct['ans_choice']
    }


# draw choices with correcr answer each questions
def draw_answer(question_no, img, keys, pos, list_answer_choice, list_diff_bubble, count):
    is_correct = False
    choice = 0
    for i, correct_ans in sorted(keys.items()):
        if correct_ans == pos and int(i) == question_no and list_diff_bubble < 40 and count == 1:
            cv2.drawContours(img, [list_answer_choice[pos - 1][2]], -1, (0, 255, 0), 3)
            is_correct = True
            choice = correct_ans
        elif correct_ans == pos and int(i) == question_no and list_diff_bubble < 40 and count > 1:
            cv2.drawContours(img, [list_answer_choice[correct_ans - 1][2]], -1, (0, 0, 255), 3)
            is_correct = False
            choice = correct_ans
        elif correct_ans == pos and int(i) == question_no and list_diff_bubble >= 40 and count == 1:
            cv2.drawContours(img, [list_answer_choice[correct_ans - 1][2]], -1, (0, 0, 255), 3)
            is_correct = False
            choice = correct_ans
        elif correct_ans == pos and int(i) == question_no and list_diff_bubble >= 40 and count > 1:
            cv2.drawContours(img, [list_answer_choice[correct_ans - 1][2]], -1, (0, 0, 255), 3)
            is_correct = False
            choice = correct_ans
        elif correct_ans != pos and int(i) == question_no and count <= 1:
            cv2.drawContours(img, [list_answer_choice[correct_ans - 1][2]], -1, (0, 0, 255), 3)
            is_correct = False
            choice = correct_ans
        elif correct_ans != pos and int(i) == question_no and count > 1:
            cv2.drawContours(img, [list_answer_choice[correct_ans - 1][2]], -1, (0, 0, 255), 3)
            is_correct = False
            choice = correct_ans
    return {
        'is_correct': is_correct,
        'ans_choice': choice
    }


def detect_circle(img_gray, num_choices):
    thresh = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2 | cv2.THRESH_OTSU)
    # cv2.imshow('adaptive', thresh)
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


def mask_std(std_cnts, img_std_id):
    stdCnts = contours.sort_contours(std_cnts, method="top-to-bottom")[0]
    list_bubbled = []
    for (q, i) in enumerate(np.arange(0, len(stdCnts), 8)):
        cnts = contours.sort_contours(stdCnts[i:i + 8])[0]
        for (j, c) in enumerate(cnts):
            mask = np.zeros(img_std_id.shape, dtype="uint8")
            cv2.drawContours(mask, [c], -1, 255, -1)
            mask = cv2.bitwise_or(img_std_id, img_std_id, mask=mask)
            total = cv2.countNonZero(mask)
            pos_choice = i + j + 1
            col = (1 if pos_choice % 8 == 1 else
                   2 if pos_choice % 8 == 2 else
                   3 if pos_choice % 8 == 3 else
                   4 if pos_choice % 8 == 4 else
                   5 if pos_choice % 8 == 5 else
                   6 if pos_choice % 8 == 6 else
                   7 if pos_choice % 8 == 7 else
                   8)
            bubbled = (total, col, pos_choice)
            list_bubbled.append(bubbled)
    print(len(list_bubbled))
    return list_bubbled


def find_std_id(list_bubbled, list_form):
    chosen_choice = 0
    list_id = []
    id_pos = -1
    total_c_form = 0
    for col in range(1, 9):
        col_choices_list = list(filter(lambda bubbles: bubbles[1] == col, list_bubbled))
        col_choices_form = list(filter(lambda bubbles: bubbles[1] == col, list_form))
        for c_form in col_choices_form:
            total_c_form += c_form[0]
        avg_c = total_c_form / 10
        for pos, bubble in enumerate(col_choices_list):
            value = bubble[0]
            diff = abs(avg_c - bubble[0])
            if chosen_choice == -1:
                chosen_choice = value
                continue
            if value > chosen_choice and diff < 40:
                chosen_choice = bubble[0]
                id_pos = pos
        chosen_choice = 1
        total_c_form = 0
        if id_pos != -1:
            list_id.append(id_pos)
    return list_id


def subtract_img(list_question_cnts, subject_img, form_img):
    blanked_img = np.zeros(form_img.shape, dtype="uint8")
    new_marker = cv2.drawContours(blanked_img, list_question_cnts, -1, (255, 255, 255), -1)
    new_marker_gray = cv2.cvtColor(new_marker, cv2.COLOR_BGR2GRAY)
    subject_gray = cv2.cvtColor(subject_img, cv2.COLOR_BGR2GRAY)
    subject_gray_blurred = cv2.GaussianBlur(subject_gray, (5, 5), 0)
    th1 = cv2.adaptiveThreshold(subject_gray_blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 71,
                                2 | cv2.THRESH_OTSU)
    new_sub = cv2.bitwise_and(new_marker_gray, th1)
    return {
        'new_sub': new_sub,
        'new_marker_gray': new_marker_gray
    }


def find_circle_contour(bound_img, amount_choices):
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
        # in order to label the contour as a question, region
        # should be sufficiently wide, sufficiently tall, and
        # have an aspect ratio approximately equal to 1
        if 15 <= w <= 50 and 15 <= h <= 50 \
                and len(approx) != 3 and len(approx) != 4 and len(approx) != 5:
            circleCnts.append(c)
    print(len(circleCnts))
    if len(circleCnts) > amount_choices:
        skip = len(circleCnts) - amount_choices
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


def main_process(form_img, subject_img, form_std_img, std_img, quiz, column, amount):
    image = cv2.imread(form_img)
    std_form = cv2.imread(form_std_img)
    subject = subject_img
    std_image = std_img
    std_image_gray = cv2.cvtColor(std_image, cv2.COLOR_BGR2GRAY)
    std_form_gray = cv2.cvtColor(std_form, cv2.COLOR_BGR2GRAY)
    stdCnts = detect_circle(std_form_gray, 80)
    check_std_cnts = detect_circle(std_image_gray, 1000)
    if len(check_std_cnts) >= 80:
        new_std_img = subtract_img(stdCnts, std_image, std_form)
        boundStdImg = cv2.drawContours(new_std_img['new_sub'], stdCnts, -1, (255, 255, 255), 1)
        circleStd = find_circle_contour(boundStdImg, 80)
        list_std_form = mask_std(stdCnts, new_std_img['new_marker_gray'])
        list_std_bubbled = mask_std(circleStd, boundStdImg)
        list_std_id = find_std_id(list_std_bubbled, list_std_form)
        if len(list_std_id) == 8:
            std_id = ''.join(map(str, list_std_id))
            print(std_id)
        else:
            std_id = 'ไม่ได้ฝนรหัสนักศึกษา'
    else:
        std_id = 'ไม่สามารถตรวจรหัสนศ.ได้ เพราะไม่สามารถ align รูปส่วนฝนรหัสนศ.ได้'
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    questionCnts = detect_circle(gray, amount*5)
    new_subject_image = subtract_img(questionCnts, subject, image)
    boundImg = cv2.drawContours(new_subject_image['new_sub'].copy(), questionCnts, -1, (255, 255, 255), 1)
    choicesCnts = find_circle_contour(boundImg, amount*5)
    list_choices_bubbled = mask_choices_bubbled(choicesCnts, boundImg, column)
    dict_c_form = mask_choices_bubbled(questionCnts, new_subject_image['new_marker_gray'], column)

    # loop for calculate score
    keys = quiz['solve']
    dict_result = {}
    for i in range(1, quiz['amount']+1):
        result = calulate_score(i, list_choices_bubbled, subject, keys, amount, column, dict_c_form)
        dict_result[str(i)] = result
    return {
        'std_id': std_id,
        'result_img': subject,
        'result': dict_result
    }
