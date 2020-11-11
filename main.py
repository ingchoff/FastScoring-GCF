import os
import tempfile
from google.cloud import storage, firestore
import firebase_admin
from firebase_admin import credentials
from PIL import Image
from exam import ImgProcess, Sift


def image_process(event, context):
    gcs = storage.Client()
    bucket = gcs.get_bucket(os.environ['CLOUD_STORAGE_BUCKET'])
    db = firestore.Client()

    print('Event ID: {}'.format(context.event_id))
    print('Event type: {}'.format(context.event_type))
    print('Bucket: {}'.format(event['bucket']))
    print('File: {}'.format(event['name']))
    print('Metageneration: {}'.format(event['metageneration']))
    print('Created: {}'.format(event['timeCreated']))
    print('Updated: {}'.format(event['updated']))

    list_folder = event['name'].split("/")
    print('{}'.format(list_folder[0]))
    subject_tmp_path = os.path.join(tempfile.gettempdir(), list_folder[1])
    form_tmp_path = os.path.join(tempfile.gettempdir(), 'form1.jpg')
    formstd_tmp_path = os.path.join(tempfile.gettempdir(), 'formStd.jpg')
    if list_folder[0] == "exam":
        form_blob = bucket.blob('form/form1.jpg')
        form_std_blob = bucket.blob('form/formStd.jpg')
        subject_blob = bucket.blob(event['name'])
        form_blob.download_to_filename(form_tmp_path)
        subject_blob.download_to_filename(subject_tmp_path)
        form_std_blob.download_to_filename(formstd_tmp_path)
        img_aligned = Sift.main_process(form_tmp_path, subject_tmp_path)
        imgstd_aligned = Sift.main_process(formstd_tmp_path, subject_tmp_path)
        result = ImgProcess.main_process(form_tmp_path, img_aligned, formstd_tmp_path, imgstd_aligned)
        result_img = Image.fromarray(result['result_img'])
        result_img.save(os.path.join(tempfile.gettempdir(), 'result.jpg'))
        result_blob = bucket.blob('result/' + 'result_' + list_folder[1])
        result_blob.upload_from_filename(os.path.join(tempfile.gettempdir(), 'result.jpg'), content_type='image/jpeg')
        doc_ref = db.collection(u'exams').document(u'RQv7EaiBnHKM3UrhNL3V')
        doc_ref.set({
            u'sid': result['std_id'],
            u'path_result': 'result/' + 'result_' + list_folder[1]
        })
        os.remove(form_tmp_path)
        os.remove(subject_tmp_path)
        os.remove(formstd_tmp_path)
        os.remove(os.path.join(tempfile.gettempdir(), 'result.jpg'))
        print('{}'.format('final'))
