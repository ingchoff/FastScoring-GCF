import cv2
import numpy as np
import os
from exam import Orb
from exam import ImgProcess
from quiz import FindAnswer
from form import CheckForm
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore


cred = credentials.Certificate('../FastScoring-ExamGrader/fastscoring-c4742ee722d9.json')
firebase_admin.initialize_app(cred)
db = firestore.client()
# quiz_ref = db.collection('quizzes').document('Hg5jta781kg8vlyj4tJ1')
quiz_ref = db.collection('quizzes').document('TalJQmTondKmhS5m5viP')
snapshot_quiz = quiz_ref.get()
form_ref = db.collection('forms').document('Ix83pdKoPCGC2zhKTpEA')
snapshot_form = form_ref.get()
data_quiz = snapshot_quiz.to_dict()
data_form = snapshot_form.to_dict()
form_tmp_path = "../FastScoring-ExamGrader/test/fullstd4.jpg"
for i in range(3, 4):
    path = "../FastScoring-ExamGrader/test/testset/angle/new/0/result/" + str(i)
    path_aligned = path + "/Aligned"
    os.mkdir(path)
    os.mkdir(path_aligned)
    subject_tmp_path = "../FastScoring-ExamGrader/test/testset/angle/new/0/" + str(i) + ".jpg"
    img_aligned = Orb.main_process(form_tmp_path, subject_tmp_path, data_form['answer_sheet_coords'], data_form['student_coords'], 'answer', 1, path_aligned)
    if not img_aligned['is_error']:
        cv2.imwrite(path + '/2.2aligned_answer.jpg', img_aligned['answer_aligned_img'])
        cv2.imwrite(path + '/2.1aligned_stu.jpg', img_aligned['stu_aligned_img'])
        result = ImgProcess.main_process(form_tmp_path, subject_tmp_path, img_aligned['answer_aligned_img'],
                                         img_aligned['stu_aligned_img'],
                                         data_quiz, data_form['column'], data_form['amount'], data_form['num_choice'],
                                         data_form['stu_column'], data_form['student_coords'],
                                         data_form['answer_sheet_coords'])
        # x_ans = int(data_form['answer_sheet_coords']['x'])
        # y_ans = int(data_form['answer_sheet_coords']['y'])
        # w_ans = int(data_form['answer_sheet_coords']['width'])
        # h_ans = int(data_form['answer_sheet_coords']['height'])
        # crop_subject_img = img_aligned['answer_aligned_img'][y_ans:y_ans + h_ans, x_ans:x_ans + w_ans]
        # result = FindAnswer.main_process(form_tmp_path, img_aligned['answer_aligned_img'], data_quiz, data_form['amount'],
        #                                  data_form['column'], data_form['answer_sheet_coords'], data_form['num_choice'])
        if not result['is_error']:
            cv2.imwrite(path + '/6result.jpg', result['result_img'])
            # print(result['score'])
            # print(result['result'])
        else:
            print(result['error_msg'])
    # x_ans = int(data_form['answer_sheet_coords']['x'])
    # y_ans = int(data_form['answer_sheet_coords']['y'])
    # w_ans = int(data_form['answer_sheet_coords']['width'])
    # h_ans = int(data_form['answer_sheet_coords']['height'])
    # form_image = cv2.imread(form_tmp_path)
    # answer_form = form_image[y_ans:y_ans + h_ans, x_ans:x_ans + w_ans]
    # checkform_result = CheckForm.main_process(answer_form, data_form['column'], data_form['amount'], "answer", data_form['num_choice'])
    # print(checkform_result['coords_choices'])
cv2.waitKey(0)
cv2.destroyAllWindows()
