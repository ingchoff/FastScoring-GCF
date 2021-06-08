import imutils
import numpy as np
from imutils import contours
import cv2
from exam import Orb, Scoring


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
def calulate_score(questions_no, answersheet, subject_img, answer_keys, amount, col, dic_c_form, point_per_clause, type_multiple):
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
        list_bubble.append(bubble[0])
        list_diff.append(diff)
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
        # กรณีที่ฝน 2 ช้อย
        if diff <= 60 and pos not in list_selected:
            list_selected.append(pos)
        # กรณีถ้าไม่ได้ฝนเลยต้องทำใหลบ
        elif diff > 60 and pos in list_selected:
            percent = (list_bubble[pos] / avg) * 100
            # print(percent)
            if percent <= 45:
                list_selected.remove(pos)
                chosen_pos = -1
            # if percent >= 45 and diff > 130:
            #     list_selected.remove(pos)
            #     chosen_pos = -1
            if percent > 45 and pos != 4 and abs(list_diff[pos]-list_diff[pos-1]) <= 42 and abs(list_diff[pos]-list_diff[pos+1]) <= 42:
                list_selected.remove(pos)
                chosen_pos = -1
            if percent > 45 and pos == 4 and abs(list_diff[pos]-list_diff[pos-1]) <= 42 and abs(list_diff[pos]-list_diff[0]) <= 42:
                list_selected.remove(pos)
                chosen_pos = -1

    print(list_selected)
    correct = draw_answer(questions_no, answer_keys, sorted(list_selected), point_per_clause, type_multiple, subject_img, answer_choice)
    return {
        'user_choice': [x+1 for x in sorted(list_selected)],
        'correct': correct['score'],
        'correct_choice': correct['ans_choice']
    }


# draw choices with correct answer each questions
def draw_answer(question_no, keys, list_pos, point_per_clause, type_multiple, img, list_answer_choice):
    list_pos = [x+1 for x in list_pos]
    is_correct = False
    choice = []
    score = 0
    # print('question:' + str(question_no))
    for i, correct_ans in sorted(keys.items()):
        if len(correct_ans) == 1 and list_pos == correct_ans and int(i) == question_no:
            # print('1')
            is_correct = True
            choice = correct_ans
            score = Scoring.main_process(correct_ans, point_per_clause, type_multiple, list_pos)
            # cv2.drawContours(img, [list_answer_choice[list_pos[0] - 1][2]], -1, (0, 255, 0), 2)
        elif len(correct_ans) > 1 and list_pos == correct_ans and int(i) == question_no:
            # print('2')
            is_correct = True
            choice = correct_ans
            score = Scoring.main_process(correct_ans, point_per_clause, type_multiple, list_pos)
            # for k in list_pos:
            #     cv2.drawContours(img, [list_answer_choice[k - 1][2]], -1, (0, 255, 0), 2)
        elif len(correct_ans) == 1 and list_pos != correct_ans and int(i) == question_no:
            # print('3')
            is_correct = False
            choice = correct_ans
            # print(list_pos)
            # cv2.drawContours(img, [list_answer_choice[correct_ans[0] - 1][2]], -1, (0, 0, 255), 2)
        elif len(correct_ans) > 1 and list_pos != correct_ans and int(i) == question_no:
            # print('4')
            is_correct = False
            choice = correct_ans
            # amount_correct = 0
            # ตรวจว่าถูกกี่ตัวเลือก
            # for j, pos in enumerate(correct_ans):
            #     if len(list_pos) != 0 and pos in list_pos:
            #         amount_correct += 1
                #     cv2.drawContours(img, [list_answer_choice[pos - 1][2]], -1, (0, 255, 0), 2)
                # else:
                #     cv2.drawContours(img, [list_answer_choice[pos - 1][2]], -1, (0, 0, 255), 2)
            score = Scoring.main_process(correct_ans, point_per_clause, type_multiple, list_pos)
        elif len(correct_ans) < 1 and list_pos != correct_ans and int(i) == question_no:
            # print('5')
            is_correct = False
            choice = correct_ans
    # print('score: ' + str(score))
    return {
        'is_correct': is_correct,
        'ans_choice': choice,
        'score': score
    }


def detect_circle(img_gray, num_choices, type_img):
    if type_img == 'std':
        value = 11
    else:
        value = 71
    thresh = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, value, 2 | cv2.THRESH_OTSU)
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


def mask_std(std_cnts, img_std_id, stu_col):
    stdCnts = contours.sort_contours(std_cnts, method="top-to-bottom")[0]
    list_bubbled = []
    for (q, i) in enumerate(np.arange(0, len(stdCnts), stu_col)):
        cnts = contours.sort_contours(stdCnts[i:i + stu_col])[0]
        for (j, c) in enumerate(cnts):
            mask = np.zeros(img_std_id.shape, dtype="uint8")
            cv2.drawContours(mask, [c], -1, 255, -1)
            mask = cv2.bitwise_or(img_std_id, img_std_id, mask=mask)
            total = cv2.countNonZero(mask)
            pos_choice = i + j + 1
            col = pos_choice % stu_col
            if col == 0:
                col = stu_col
            bubbled = (total, col, pos_choice)
            list_bubbled.append(bubbled)
    # print(len(list_bubbled))
    return list_bubbled


def find_std_id(list_bubbled, list_form, stu_col):
    list_id = []
    for col in range(1, stu_col+1):
        list_selected = []
        chosen_choice = None
        chosen_pos = 0
        total_c_form = 0
        list_diff = []
        list_bubble = []
        print('col: ' + str(col))
        col_choices_list = list(filter(lambda bubbles: bubbles[1] == col, list_bubbled))
        col_choices_form = list(filter(lambda bubbles: bubbles[1] == col, list_form))
        for c_form in col_choices_form:
            total_c_form += c_form[0]
        avg_c = total_c_form / 10
        for bubble in col_choices_list:
            diff = (abs(avg_c - bubble[0]))
            list_bubble.append(bubble[0])
            list_diff.append(diff)
        for pos, bubble in enumerate(col_choices_list):
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
        for pos, diff in enumerate(list_diff):
            if diff <= 60 and pos not in list_selected:
                list_selected.append(pos)
            elif diff > 60 and pos in list_selected:
                percent = (list_bubble[pos] / avg_c) * 100
                print(percent)
                if percent < 45:
                    list_selected.remove(pos)
                if percent >= 45 and pos != 9 and abs(list_diff[pos] - list_diff[pos - 1]) <= 42 and abs(
                        list_diff[pos] - list_diff[pos + 1]) <= 42:
                    list_selected.remove(pos)
                if percent >= 45 and pos == 9 and abs(list_diff[pos] - list_diff[pos - 1]) <= 42 and abs(
                        list_diff[pos] - list_diff[0]) <= 42:
                    list_selected.remove(pos)
        print(list_selected)
        for pos, stu_id in enumerate(list_selected):
            if pos == 0:
                list_id.append(stu_id)
    return list_id


def subtract_img(list_question_cnts, subject_gray, form_img, type_img):
    if type_img == 'answer':
        value = 71
    else:
        value = 71
    blanked_img = np.zeros(form_img.shape, dtype="uint8")
    new_marker = cv2.drawContours(blanked_img, list_question_cnts, -1, (255, 255, 255), -1)
    new_marker_gray = cv2.cvtColor(new_marker, cv2.COLOR_BGR2GRAY)
    subject_gray_blurred = cv2.GaussianBlur(subject_gray, (5, 5), 0)
    th1 = cv2.adaptiveThreshold(subject_gray_blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, value,
                                2 | cv2.THRESH_OTSU)
    # cv2.imwrite('3new_marker.jpg', new_marker)
    # cv2.imwrite('3answer-threshold.jpg', th1)
    new_sub = cv2.bitwise_and(new_marker_gray, th1)
    # cv2.imwrite('3answer-subtract.jpg', new_sub)
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
            # cv2.imwrite('ex_form' + str(i + j + 1) + '.jpg', mask)
            # apply the mask to the thresholded image, then
            # count the number of non-zero pixels in the
            # bubble area
            mask = cv2.bitwise_or(bound_img, bound_img, mask=mask)
            # cv2.imwrite('ex' + str(i + j + 1) + '.jpg', mask)
            total = cv2.countNonZero(mask)
            # if the current total has a larger number of total
            # non-zero pixels, then we are examining the currently
            # bubbled-in answer
            bubbled = (total, i + j + 1, c)
            list_bubbled.append(bubbled)
    return list_bubbled


def main_process(form_img, subject_tmp_path, subject_img, std_img, quiz, column, amount, num_choice, stu_col, stu_coords, answer_coords):
    image = cv2.imread(form_img)
    x_ans = int(answer_coords['x'])
    y_ans = int(answer_coords['y'])
    w_ans = int(answer_coords['width'])
    h_ans = int(answer_coords['height'])
    x_stu = int(stu_coords['x'])
    y_stu = int(stu_coords['y'])
    w_stu = int(stu_coords['width'])
    h_stu = int(stu_coords['height'])
    answer_form = image[y_ans:y_ans + h_ans, x_ans:x_ans + w_ans]
    std_form = image[y_stu:y_stu + h_stu, x_stu:x_stu + w_stu]
    subject = subject_img
    std_image = std_img
    std_image_gray = cv2.cvtColor(std_image, cv2.COLOR_BGR2GRAY)
    std_form_gray = cv2.cvtColor(std_form, cv2.COLOR_BGR2GRAY)
    check_std_cnts = detect_circle(std_image_gray, 1000, 'std')
    std_id = ''
    if len(check_std_cnts) >= 10*stu_col:
        stdCnts = detect_circle(std_form_gray, 10*stu_col, 'std')
        new_std_img = subtract_img(stdCnts, std_image_gray, std_form, 'std')
        boundStdImg = cv2.drawContours(new_std_img['new_sub'], stdCnts, -1, (255, 255, 255), 1)
        circleStd = find_circle_contour(boundStdImg, 10*stu_col)
        list_std_form = mask_std(stdCnts, new_std_img['new_marker_gray'], stu_col)
        list_std_bubbled = mask_std(circleStd, boundStdImg, stu_col)
        list_std_id = find_std_id(list_std_bubbled, list_std_form, stu_col)
        if len(list_std_id) == stu_col:
            std_id = ''.join(map(str, list_std_id))
            print(std_id)
        elif 1 <= stu_col - len(list_std_id) <= 7:
            img_aligned = Orb.main_process(form_img, subject_tmp_path, answer_coords, stu_coords, 'answer', 2, '')
            std_image = img_aligned['stu_aligned_img']
            subject = img_aligned['answer_aligned_img']
            std_image_gray = cv2.cvtColor(std_image, cv2.COLOR_BGR2GRAY)
            std_form_gray = cv2.cvtColor(std_form, cv2.COLOR_BGR2GRAY)
            check_std_cnts = detect_circle(std_image_gray, 1000, 'std')
            if len(check_std_cnts) >= 10 * stu_col:
                stdCnts = detect_circle(std_form_gray, 10 * stu_col, 'std')
                new_std_img = subtract_img(stdCnts, std_image_gray, std_form, 'std')
                boundStdImg = cv2.drawContours(new_std_img['new_sub'], stdCnts, -1, (255, 255, 255), 1)
                circleStd = find_circle_contour(boundStdImg, 10 * stu_col)
                list_std_form = mask_std(stdCnts, new_std_img['new_marker_gray'], stu_col)
                list_std_bubbled = mask_std(circleStd, boundStdImg, stu_col)
                list_std_id = find_std_id(list_std_bubbled, list_std_form, stu_col)
                if len(list_std_id) == stu_col:
                    std_id = ''.join(map(str, list_std_id))
                    print(std_id)
                else:
                    std_id = 'ไม่ได้ฝนรหัสนักศึกษา'
        else:
            std_id = 'ไม่ได้ฝนรหัสนักศึกษา'
    else:
        std_id = 'ไม่สามารถตรวจรหัสนศ.ได้เพราะ align รูปส่วนฝนรหัสนศ.ได้ไม่ถูกต้อง'
    coords_choices = {}
    gray = cv2.cvtColor(answer_form, cv2.COLOR_BGR2GRAY)
    subject_gray = cv2.cvtColor(subject, cv2.COLOR_BGR2GRAY)
    check_ans_cnts = detect_circle(subject_gray, 1000, 'answer')
    if len(check_ans_cnts) >= amount*num_choice:
        questionCnts = detect_circle(gray, amount*num_choice, 'answer')
        circle_choices = detect_circle(subject_gray, amount * num_choice, 'answer')
        new_subject_image = subtract_img(questionCnts, subject_gray, answer_form, 'answer')
        boundImg = cv2.drawContours(new_subject_image['new_sub'].copy(), questionCnts, -1, (255, 255, 255), 1)
        # cv2.imwrite('4boundImg.jpg', boundImg)
        choicesCnts = find_circle_contour(boundImg, amount*num_choice)
        list_choices_bubbled = mask_choices_bubbled(choicesCnts, boundImg, column)
        # dict_c_form = mask_choices_bubbled(questionCnts, new_subject_image['new_marker_gray'], column)
        choices_cnts = contours.sort_contours(circle_choices, method="top-to-bottom")[0]
        for (q, i) in enumerate(np.arange(0, len(choices_cnts), int(num_choice) * int(column))):
            cnts = contours.sort_contours(choices_cnts[i:i + int(num_choice) * int(column)])[0]
            for (j, c) in enumerate(cnts):
                (x, y, w, h) = cv2.boundingRect(c)
                coords_choices[str(i + j + 1)] = {'x': x, 'y': y}
        # loop for calculate score
        keys = quiz['solve']
        dict_result = {}
        score = 0
        # for i in range(1, quiz['amount']+1):
        #     print('question: ' + str(i))
        #     result = calulate_score(i, list_choices_bubbled, subject, keys, amount, column, dict_c_form, quiz['point_per_clause'], quiz['multiple_choice'])
        #     if result['correct']:
        #         score += result['correct']
        #     dict_result[str(i)] = result
        return {
            'is_error': False,
            'std_id': std_id,
            'result_img': subject,
            'result': dict_result,
            'score': score,
            'coords_choices': coords_choices
        }
    else:
        return {
            'std_id': std_id,
            'error_msg': 'ไม่สามารถตรวจข้อสอบได้เพราะ align รูปส่วนฝนคำตอบได้ไม่ถูกต้อง',
            'is_error': True
        }
