import os
import tempfile
from google.cloud import storage, firestore_v1
import firebase_admin
from firebase_admin import credentials
from PIL import Image
import cv2
import json
from exam import ImgProcess, Sift
from form import CheckForm
from quiz import FindAnswer
from exam import Orb, Scoring

gcs = storage.Client()
bucket = gcs.get_bucket(os.environ['CLOUD_STORAGE_BUCKET'])
db = firestore_v1.Client()


def img_process(event, context):

    print('Event type: {}'.format(context.event_type))
    print('File: {}'.format(event['name']))
    print('Created: {}'.format(event['timeCreated']))
    print('Updated: {}'.format(event['updated']))

    list_folder = event['name'].split("/")
    print('{}'.format(list_folder[0]))
    if list_folder[0] == "exams":
        list_filename_exam = list_folder[3].split('_')
        # set firestore
        exam_ref = db.collection('exams').document(list_filename_exam[0])
        exam_snapshot = exam_ref.get()
        data_exam = exam_snapshot.to_dict()
        quiz_snapshot = data_exam['quiz'].get()
        data_quiz = quiz_snapshot.to_dict()
        form_snapshot = data_quiz['form'].get()
        data_form = form_snapshot.to_dict()
        list_fid = data_form['analysed_answersheet_path'].split('/')
        fid = list_fid[2].split('_')[1]
        answer_sheet_path = '/forms/' + data_form['owner'] + '/' + fid + '_form.jpg'
        list_form_path = answer_sheet_path.split('/')
        # set tmp path for download images
        subject_tmp_path = os.path.join(tempfile.gettempdir(), list_folder[3])
        form_tmp_path = os.path.join(tempfile.gettempdir(), list_form_path[3])
        # set blob destination file to download file
        form_blob = bucket.blob(list_form_path[1] + '/' + list_form_path[2] + '/' + list_form_path[3])
        subject_blob = bucket.blob(event['name'])
        form_blob.download_to_filename(form_tmp_path)
        subject_blob.download_to_filename(subject_tmp_path)
        exam_ref.set({
            'status': 'aligning'
        }, merge=True)
        img_aligned = Orb.main_process(form_tmp_path, subject_tmp_path, data_form['answer_sheet_coords'], data_form['student_coords'], 'answer', 1)
        # img_aligned = Sift.main_process(form_tmp_path, subject_tmp_path, 'answer')
        # imgstd_aligned = Sift.main_process(formstd_tmp_path, subject_tmp_path, 'std')
        if not img_aligned['is_error']:
            exam_ref.set({
                'status': 'scoring'
            }, merge=True)
            result = ImgProcess.main_process(form_tmp_path, subject_tmp_path, img_aligned['answer_aligned_img'], img_aligned['stu_aligned_img'],
                                             data_quiz, data_form['column'], data_form['amount'], data_form['num_choice'],
                                             data_form['stu_column'], data_form['student_coords'], data_form['answer_sheet_coords'])
            if not result['is_error']:
                bound_img_rgb = cv2.cvtColor(result['result_img'], cv2.COLOR_BGR2RGB)
                result_img = Image.fromarray(bound_img_rgb)
                result_img.save(os.path.join(tempfile.gettempdir(), 'result.jpg'))
                result_blob = bucket.blob('result/' + list_folder[1] + '/result_' + list_folder[3])
                exam_ref.set({
                    'status': 'uploading result'
                }, merge=True)
                result_blob.upload_from_filename(os.path.join(tempfile.gettempdir(), 'result.jpg'), content_type='image/jpeg')
                exam_ref.set({
                    'sid': result['std_id'],
                    'path_result': 'result/' + list_folder[1] + '/result_' + list_folder[3],
                    'status': 'done',
                    'result': result['result'],
                    'score': result['score'],
                    'coords_choices': result['coords_choices']
                }, merge=True)
                os.remove(os.path.join(tempfile.gettempdir(), 'result.jpg'))
            else:
                exam_ref.set({
                    'sid': result['std_id'],
                    'status': 'error',
                    'error_msg': result['error_msg']
                }, merge=True)
        else:
            exam_ref.set({
                'status': 'error',
                'error_msg': 'ไม่สามารถ align รูปส่วนฝนรหัสนศ.และส่วนฝนคำตอบได้'
            }, merge=True)
        os.remove(form_tmp_path)
        os.remove(subject_tmp_path)
        # os.remove(formstd_tmp_path)
        print('{}'.format('final'))
    if list_folder[0] == "quizzes":
        list_filename = list_folder[2].split('_')
        print('{}'.format(list_filename[0]))
        quiz_id = list_filename[0]
        quiz_type = list_filename[1]
        file_name = list_filename[1]
        if quiz_id != "analysed":
            find_solve(list_folder, quiz_id, quiz_type, file_name)


# def check_form(list_folder, fid, form_type, filename):
#     # set up firestore
#     form_ref = db.collection('forms').document(fid)
#     snapshot_form = form_ref.get()
#     data_form = snapshot_form.to_dict()
#     # set tmp path for download images
#     form_tmp_path = os.path.join(tempfile.gettempdir(), list_folder[2])
#     if form_type == "answersheet.png":
#         # set blob destination file to download file
#         form_blob = bucket.blob(list_folder[0] + '/' + list_folder[1] + '/' + list_folder[2])
#         form_blob.download_to_filename(form_tmp_path)
#         form_ref.set({
#             'answer_status': 'analysing'
#         }, merge=True)
#         is_available = CheckForm.main_process(form_tmp_path, data_form['column'], data_form['amount'], form_type)
#         if is_available['available']:
#             bound_img_rgb = cv2.cvtColor(is_available['bound_img'], cv2.COLOR_BGR2RGB)
#             analysed_img = Image.fromarray(bound_img_rgb)
#             analysed_img.save(os.path.join(tempfile.gettempdir(), 'analysed.jpg'))
#             analysed_blob = bucket.blob('forms/' + list_folder[1] + '/analysed_' + fid + '_' + filename)
#             analysed_blob.upload_from_filename(os.path.join(tempfile.gettempdir(), 'analysed.jpg'),
#                                                content_type='image/jpeg')
#             form_ref.set({
#                 'answer_status': 'pass',
#                 'analysed_answersheet_path': 'forms/' + list_folder[1] + '/analysed_' + fid + '_' + filename
#             }, merge=True)
#         else:
#             form_ref.set({
#                 'error_ans_msg': 'Not Compatible',
#                 'answer_status': 'error'
#             }, merge=True)
#         os.remove(form_tmp_path)
#     if form_type == "student.png":
#         # set blob destination file to download file
#         form_blob = bucket.blob(list_folder[0] + '/' + list_folder[1] + '/' + list_folder[2])
#         form_blob.download_to_filename(form_tmp_path)
#         form_ref.set({
#             'stu_status': 'analysing'
#         }, merge=True)
#         is_available = CheckForm.main_process(form_tmp_path, data_form['stu_column'], data_form['amount'], form_type)
#         if is_available['available']:
#             bound_img_rgb = cv2.cvtColor(is_available['bound_img'], cv2.COLOR_BGR2RGB)
#             analysed_img = Image.fromarray(bound_img_rgb)
#             analysed_img.save(os.path.join(tempfile.gettempdir(), 'analysed.jpg'))
#             analysed_blob = bucket.blob('forms/' + list_folder[1] + '/analysed_' + fid + '_' + filename)
#             analysed_blob.upload_from_filename(os.path.join(tempfile.gettempdir(), 'analysed.jpg'),
#                                                content_type='image/jpeg')
#             form_ref.set({
#                 'stu_status': 'pass',
#                 'analysed_stu_path': 'forms/' + list_folder[1] + '/analysed_' + fid + '_' + filename
#             }, merge=True)
#         else:
#             form_ref.set({
#                 'error_stu_msg': 'Not Compatible',
#                 'stu_status': 'error'
#             }, merge=True)
#         os.remove(form_tmp_path)


def find_solve(list_folder, qid, quiz_type, filename):
    # set up firestore
    quiz_ref = db.collection('quizzes').document(qid)
    snapshot_quiz = quiz_ref.get()
    data_quiz = snapshot_quiz.to_dict()
    snapshot_form = data_quiz['form'].get()
    data_form = snapshot_form.to_dict()
    # set tmp path for download images
    list_fid = data_form['analysed_answersheet_path'].split('/')
    fid = list_fid[2].split('_')[1]
    form_path = '/forms/' + data_form['owner'] + '/' + fid + '_form.jpg'
    list_form_path = form_path.split('/')
    # list_form_path = data_form['answer_sheet_path'].split('/')
    solve_tmp_path = os.path.join(tempfile.gettempdir(), filename)
    form_tmp_path = os.path.join(tempfile.gettempdir(), fid + '_answersheet.png')
    # set blob destination file to download file
    form_blob = bucket.blob(list_form_path[1] + '/' + list_form_path[2] + '/' + list_form_path[3])
    form_blob.download_to_filename(form_tmp_path)
    solve_blob = bucket.blob(list_folder[0] + '/' + list_folder[1] + '/' + list_folder[2])
    solve_blob.download_to_filename(solve_tmp_path)
    quiz_ref.set({
        'solution_status': 'process',
        'detail': 'aligning'
    }, merge=True)
    img_aligned = Orb.main_process(form_tmp_path, solve_tmp_path, data_form['answer_sheet_coords'], data_form['student_coords'], 'answer', 1)
    # img_aligned = Sift.main_process(form_tmp_path, solve_tmp_path, 'answer')
    if 'answer_aligned_img' in img_aligned:
        quiz_ref.set({
            'solution_status': 'process',
            'detail': 'analysing',
            'solve': {}
        }, merge=True)
        result_solve = FindAnswer.main_process(form_tmp_path, img_aligned['answer_aligned_img'], data_quiz, data_form['amount'],
                                               data_form['column'], data_form['answer_sheet_coords'], data_form['num_choice'])
        if 'img_solve' in result_solve:
            bound_img_rgb = cv2.cvtColor(result_solve['img_solve'], cv2.COLOR_BGR2RGB)
            analysed_img = Image.fromarray(bound_img_rgb)
            analysed_img.save(os.path.join(tempfile.gettempdir(), 'analysed_solve.jpg'))
            analysed_blob = bucket.blob('quizzes/' + list_folder[1] + '/analysed_' + qid + '_' + filename)
            analysed_blob.upload_from_filename(os.path.join(tempfile.gettempdir(), 'analysed_solve.jpg'),
                                               content_type='image/jpeg')
            quiz_ref.update({
                'solve': result_solve['result_solve']
            })
            quiz_ref.set({
                'solution_status': 'finish',
                'analysed_solution_path': 'quizzes/' + list_folder[1] + '/analysed_' + qid + '_' + filename
            }, merge=True)
            quiz_ref.update({
                'detail': firestore_v1.DELETE_FIELD
            })
        else:
            quiz_ref.set({
                'solution_status': 'error',
                'detail': result_solve['error_msg']
            }, merge=True)
    else:
        quiz_ref.set({
            'solution_status': 'error',
            'detail': img_aligned['error_msg']
        }, merge=True)
    os.remove(form_tmp_path)
    os.remove(solve_tmp_path)


def coords(data, context):
    """ Triggered by a change to a Firestore document.
    Args:
        data (dict): The event payload.
        context (google.cloud.functions.Context): Metadata for the event.
    """

    print('Event type: {}'.format(context.resource))
    print('{}'.format(data))
    list_resource = context.resource.split("/")
    if list_resource[5] == 'forms':
        form_ref = db.collection('forms').document(list_resource[6])
        # snapshot_form = form_ref.get()
        # data_form = snapshot_form.to_dict()
        if not data['value']['fields'].get('answer_status'):
            return
        print(data['value']['fields']['answer_status'])
        if data['value']['fields']['answer_status']['stringValue'] == 'pending' \
                or data['value']['fields']['answer_status']['stringValue'] == 'resend':
            print(data['value']['fields']['num_choice']['integerValue'])
            answer_coords = data['value']['fields']['answer_sheet_coords']['mapValue']['fields']
            form_tmp_path = os.path.join(tempfile.gettempdir(), 'form_tmp.jpg')
            owner = data['value']['fields']['owner']['stringValue']
            form_blob = bucket.blob('forms' + '/' + owner + '/' + list_resource[6] + '_form.jpg')
            form_blob.download_to_filename(form_tmp_path)
            x = 0
            y = 0
            h = 0
            w = 0
            for key in answer_coords['x']:
                if key == 'doubleValue':
                    x = int(answer_coords['x']['doubleValue'])
                else:
                    x = int(answer_coords['x']['integerValue'])
            for key in answer_coords['y']:
                if key == 'doubleValue':
                    y = int(answer_coords['y']['doubleValue'])
                else:
                    y = int(answer_coords['y']['integerValue'])
            for key in answer_coords['width']:
                if key == 'doubleValue':
                    w = int(answer_coords['width']['doubleValue'])
                else:
                    w = int(answer_coords['width']['integerValue'])
            for key in answer_coords['height']:
                if key == 'doubleValue':
                    h = int(answer_coords['height']['doubleValue'])
                else:
                    h = int(answer_coords['height']['integerValue'])
            form_full = cv2.imread(form_tmp_path)
            crop_img_answer = form_full[y:y + h, x:x + w]
            form_ref.set({
                'answer_status': 'analysing'
            }, merge=True)
            is_available_answer = CheckForm.main_process(crop_img_answer, data['value']['fields']['column']['integerValue'],
                                                         data['value']['fields']['amount']['integerValue'], 'answer', data['value']['fields']['num_choice']['integerValue'])
            if is_available_answer['available']:
                bound_img_rgb = cv2.cvtColor(is_available_answer['bound_img'], cv2.COLOR_BGR2RGB)
                analysed_img = Image.fromarray(bound_img_rgb)
                analysed_img.save(os.path.join(tempfile.gettempdir(), 'analysed.jpg'))
                analysed_blob = bucket.blob('forms/' + owner + '/analysed_' + list_resource[6] + '_' + 'answersheet.jpg')
                analysed_blob.upload_from_filename(os.path.join(tempfile.gettempdir(), 'analysed.jpg'),
                                                   content_type='image/jpeg')
                print(is_available_answer['coords_choices'])
                form_ref.set({
                    'answer_status': 'pass',
                    'analysed_answersheet_path': 'forms/' + data['value']['fields']['owner']['stringValue'] + '/analysed_' + list_resource[6] + '_' + 'answersheet.jpg',
                    'coords_choices': is_available_answer['coords_choices']
                }, merge=True)
            else:
                form_ref.set({
                    'error_ans_msg': 'Not Compatible',
                    'answer_status': 'error'
                }, merge=True)
            os.remove(form_tmp_path)
        if not data['value']['fields'].get('stu_status'):
            return
        if data['value']['fields']['stu_status']['stringValue'] == 'pending' \
                or data['value']['fields']['stu_status']['stringValue'] == 'resend':
            stu_coords = data['value']['fields']['student_coords']['mapValue']['fields']
            x = 0
            y = 0
            h = 0
            w = 0
            for key in stu_coords['x']:
                if key == 'doubleValue':
                    x = int(stu_coords['x']['doubleValue'])
                else:
                    x = int(stu_coords['x']['integerValue'])
            for key in stu_coords['y']:
                if key == 'doubleValue':
                    y = int(stu_coords['y']['doubleValue'])
                else:
                    y = int(stu_coords['y']['integerValue'])
            for key in stu_coords['width']:
                if key == 'doubleValue':
                    w = int(stu_coords['width']['doubleValue'])
                else:
                    w = int(stu_coords['width']['integerValue'])
            for key in stu_coords['height']:
                if key == 'doubleValue':
                    h = int(stu_coords['height']['doubleValue'])
                else:
                    h = int(stu_coords['height']['integerValue'])
            form_tmp_path = os.path.join(tempfile.gettempdir(), 'form_tmp.jpg')
            owner = data['value']['fields']['owner']['stringValue']
            form_blob = bucket.blob('forms' + '/' + owner + '/' + list_resource[6] + '_form.jpg')
            form_blob.download_to_filename(form_tmp_path)
            form_full = cv2.imread(form_tmp_path)
            crop_img_stu = form_full[y:y + h, x:x + w]
            form_ref.set({
                'stu_status': 'analysing'
            }, merge=True)
            is_available_stu = CheckForm.main_process(crop_img_stu, data['value']['fields']['stu_column']['integerValue'],
                                                      data['value']['fields']['amount']['integerValue'], 'stu', 0)
            if is_available_stu['available']:
                bound_img_rgb = cv2.cvtColor(is_available_stu['bound_img'], cv2.COLOR_BGR2RGB)
                analysed_img = Image.fromarray(bound_img_rgb)
                analysed_img.save(os.path.join(tempfile.gettempdir(), 'analysed.jpg'))
                analysed_blob = bucket.blob('forms/' + owner + '/analysed_' + list_resource[6] + '_' + 'student.jpg')
                analysed_blob.upload_from_filename(os.path.join(tempfile.gettempdir(), 'analysed.jpg'),
                                                   content_type='image/jpeg')
                form_ref.set({
                    'stu_status': 'pass',
                    'analysed_stu_path': 'forms/' + data['value']['fields']['owner']['stringValue'] + '/analysed_' + list_resource[6] + '_' + 'student.jpg'
                }, merge=True)
            else:
                form_ref.set({
                    'error_ans_msg': 'Not Compatible',
                    'stu_status': 'error'
                }, merge=True)
            os.remove(form_tmp_path)
    if list_resource[5] == 'quizzes':
        quiz_ref = db.collection('quizzes').document(list_resource[6])
        query_exam_ref = db.collection('exams').where("quiz", "==", quiz_ref)
        if not data['value']['fields'].get('multiple_choice'):
            return
        type_multiple = data['value']['fields']['multiple_choice']['stringValue']
        print(type_multiple)
        point_per_clause = data['value']['fields']['point_per_clause']['integerValue']
        print(point_per_clause)
        # exam_ref.set({
        #     'status': 'scoring'
        # })
        # Scoring.main_process()

