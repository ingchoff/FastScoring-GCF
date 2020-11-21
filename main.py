import os
import tempfile
from google.cloud import storage, firestore_v1
import firebase_admin
from firebase_admin import credentials
from PIL import Image
import cv2
from exam import ImgProcess, Sift
from form import CheckForm

gcs = storage.Client()
bucket = gcs.get_bucket(os.environ['CLOUD_STORAGE_BUCKET'])
db = firestore_v1.Client()


def image_process(event, context):

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
        form_snapshot = data_exam['form'].get()
        data_form = form_snapshot.to_dict()
        quiz_snapshot = data_exam['quiz'].get()
        data_quiz = quiz_snapshot.to_dict()
        list_filename_form = data_form['path'].split('/')
        # set tmp path for download images
        subject_tmp_path = os.path.join(tempfile.gettempdir(), list_folder[3])
        form_tmp_path = os.path.join(tempfile.gettempdir(), list_filename_form[2])
        formstd_tmp_path = os.path.join(tempfile.gettempdir(), 'formStd.jpg')
        # set blob destination file to download file
        form_blob = bucket.blob(data_form['path'])
        form_std_blob = bucket.blob('form/' + list_folder[1] + '/formStd.jpg')
        subject_blob = bucket.blob(event['name'])
        form_blob.download_to_filename(form_tmp_path)
        subject_blob.download_to_filename(subject_tmp_path)
        form_std_blob.download_to_filename(formstd_tmp_path)
        exam_ref.set({
            'status': 'aligning'
        }, merge=True)
        img_aligned = Sift.main_process(form_tmp_path, subject_tmp_path)
        imgstd_aligned = Sift.main_process(formstd_tmp_path, subject_tmp_path)
        exam_ref.set({
            'status': 'scoring'
        }, merge=True)
        result = ImgProcess.main_process(form_tmp_path, img_aligned, formstd_tmp_path, imgstd_aligned,
                                         data_quiz)
        result_img = Image.fromarray(result['result_img'])
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
            'result': result['result']
        }, merge=True)
        os.remove(form_tmp_path)
        os.remove(subject_tmp_path)
        os.remove(formstd_tmp_path)
        os.remove(os.path.join(tempfile.gettempdir(), 'result.jpg'))
        print('{}'.format('final'))
    if list_folder[0] == "forms":
        list_filename = list_folder[2].split('_')
        print('{}'.format(list_filename))
        form_id = list_filename[0]
        type_form = list_filename[1]
        file_name = list_filename[1]
        if form_id != "analysed":
            print("{}".format('start checking'))
            check_form(list_folder, form_id, type_form, file_name)


def check_form(list_folder, fid, form_type, filename):
    # set up firestore
    form_ref = db.collection('forms').document(fid)
    snapshot_form = form_ref.get()
    data_form = snapshot_form.to_dict()
    # set tmp path for download images
    form_tmp_path = os.path.join(tempfile.gettempdir(), list_folder[2])
    if form_type == "answersheet.png":
        # set blob destination file to download file
        form_blob = bucket.blob(list_folder[0] + '/' + list_folder[1] + '/' + list_folder[2])
        form_blob.download_to_filename(form_tmp_path)
        form_ref.set({
            'answer_status': 'analysing'
        }, merge=True)
        is_available = CheckForm.main_process(form_tmp_path, data_form['column'], data_form['amount'], form_type)
        if is_available['available']:
            bound_img_rgb = cv2.cvtColor(is_available['bound_img'], cv2.COLOR_BGR2RGB)
            analysed_img = Image.fromarray(bound_img_rgb)
            analysed_img.save(os.path.join(tempfile.gettempdir(), 'analysed.jpg'))
            analysed_blob = bucket.blob('forms/' + list_folder[1] + '/analysed_' + fid + '_' + filename)
            analysed_blob.upload_from_filename(os.path.join(tempfile.gettempdir(), 'analysed.jpg'),
                                               content_type='image/jpeg')
            form_ref.set({
                'answer_status': 'pass',
                'analysed_answersheet_path': 'forms/' + list_folder[1] + '/analysed_' + fid + '_' + filename
            }, merge=True)
        else:
            form_ref.set({
                'error_ans_msg': 'Not Compatible',
                'answer_status': 'error'
            }, merge=True)
        os.remove(form_tmp_path)
    if form_type == "student.png":
        # set blob destination file to download file
        form_blob = bucket.blob(list_folder[0] + '/' + list_folder[1] + '/' + list_folder[2])
        form_blob.download_to_filename(form_tmp_path)
        form_ref.set({
            'stu_status': 'analysing'
        }, merge=True)
        is_available = CheckForm.main_process(form_tmp_path, data_form['stu_column'], data_form['amount'], form_type)
        if is_available['available']:
            bound_img_rgb = cv2.cvtColor(is_available['bound_img'], cv2.COLOR_BGR2RGB)
            analysed_img = Image.fromarray(bound_img_rgb)
            analysed_img.save(os.path.join(tempfile.gettempdir(), 'analysed.jpg'))
            analysed_blob = bucket.blob('forms/' + list_folder[1] + '/analysed_' + fid + '_' + filename)
            analysed_blob.upload_from_filename(os.path.join(tempfile.gettempdir(), 'analysed.jpg'),
                                               content_type='image/jpeg')
            form_ref.set({
                'stu_status': 'pass',
                'analysed_stu_path': 'forms/' + list_folder[1] + '/analysed_' + fid + '_' + filename
            }, merge=True)
        else:
            form_ref.set({
                'error_stu_msg': 'Not Compatible',
                'stu_status': 'error'
            }, merge=True)
