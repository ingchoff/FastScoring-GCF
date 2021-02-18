import imutils
import numpy as np
from imutils import contours
import cv2


def find_contour_circle(form_img, num_choices):
    thresh = cv2.adaptiveThreshold(form_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2 | cv2.THRESH_OTSU)
    contour = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contour = imutils.grab_contours(contour)
    contour = sorted(contour, key=cv2.contourArea, reverse=True)
    list_circle = []
    for c in contour:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.03 * peri, True)
        (x, y, w, h) = cv2.boundingRect(c)
        if 10 <= w <= 50 and 10 <= h <= 50 \
                and len(approx) != 3 and len(approx) != 4 and len(approx) != 5:
            list_circle.append(c)
    print(len(list_circle))
    if len(list_circle) > num_choices:
        skip = len(list_circle) - num_choices
        list_circle_after = list_circle[:-int(skip)]
    else:
        list_circle_after = list_circle
    print(len(list_circle_after))
    return {'list_circle': list_circle,
            'list_circle_after': list_circle_after}


def main_process(form_img, column, amount, form_type, num_choice):
    form = form_img
    if form_type == "answer":
        row = int(int(amount) / int(column))
        form_gray = cv2.cvtColor(form, cv2.COLOR_BGR2GRAY)
        # still fixed choice each question = 5
        print(num_choice)
        circle_cnts = find_contour_circle(form_gray, int(num_choice)*int(column)*int(row))
        choices_cnts = contours.sort_contours(circle_cnts['list_circle_after'], method="top-to-bottom")[0]
        cv2.drawContours(form, choices_cnts, -1, (0, 0, 255), 2)
        # check this form is compatible with system.
        if len(circle_cnts['list_circle_after']) == int(amount)*int(num_choice):
            return {
                'bound_img': form,
                'available': True
            }
        else:
            return {
                'available': False
            }
    else:
        row = 10
        form_gray = cv2.cvtColor(form, cv2.COLOR_BGR2GRAY)
        circle_cnts = find_contour_circle(form_gray, row*int(column))
        choices_cnts = contours.sort_contours(circle_cnts['list_circle_after'], method="top-to-bottom")[0]
        cv2.drawContours(form, choices_cnts, -1, (0, 0, 255), 2)
        if len(circle_cnts['list_circle_after']) == row*int(column):
            return {
                'bound_img': form,
                'available': True
            }
        else:
            return {
                'available': False
            }
