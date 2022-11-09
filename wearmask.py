import os
import sys
import argparse
import numpy as np
import cv2
import random
from PIL import Image, ImageFile
__version__ = '0.3.0'

IMAGE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mask_images')
cloth_PATH = os.path.join(IMAGE_DIR, 'cloth.png')
surgical_white_PATH = os.path.join(IMAGE_DIR, 'surgical.png')
surgical_blue_IMAGE_PATH = os.path.join(IMAGE_DIR, 'surgical_blue.png')
surgical_green_IMAGE_PATH = os.path.join(IMAGE_DIR, 'surgical_green.png')
KN95_IMAGE_PATH = os.path.join(IMAGE_DIR, 'KN95.png')


def mask_select():

    num = random.randint(1,100)
    if 0 <= num % 100 < 30:
        return cloth_PATH
    elif 30 <= num % 100 < 60:
        return surgical_blue_IMAGE_PATH
    elif 60 <= num % 100 < 90:
        return surgical_white_PATH
    elif 90 <= num % 100 < 95:
        return surgical_green_IMAGE_PATH
    elif 95 <= num % 100 < 100:
        return KN95_IMAGE_PATH
    else:
        print("mask select error")


def rect_to_bbox(rect):
    """获得人脸矩形的坐标信息"""
    # print(rect)
    x = rect[3]
    y = rect[0]
    w = rect[1] - x
    h = rect[2] - y
    return (x, y, w, h)


def cli(pic_path,save_pic_path, mask_path):
    parser = argparse.ArgumentParser(description='Wear a face mask in the given picture.')
    parser.add_argument('--model', default='hog', choices=['hog', 'cnn'], help='Which face detection model to use.')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--black', action='store_true', help='Wear black mask')
    group.add_argument('--blue', action='store_true', help='Wear blue mask')
    group.add_argument('--red', action='store_true', help='Wear red mask')
    args = parser.parse_args()

    if not os.path.exists(pic_path):
        print(f'Picture {pic_path} not exists.')
        sys.exit(1)

    if not mask_path:
        mask_path = mask_select()

    FaceMasker(pic_path, mask_path, True, 'hog',save_pic_path).mask()


class FaceMasker:
    KEY_FACIAL_FEATURES = ('nose_bridge', 'chin')

    def __init__(self, face_path, mask_path, show=False, model='hog',save_path = ''):
        self.face_path = face_path
        self.mask_path = mask_path
        self.save_path = save_path
        self.show = show
        self.model = model
        self._face_img: ImageFile = None
        self._mask_img: ImageFile = None

    def mask(self):
        import face_recognition

        face_image_np = face_recognition.load_image_file(self.face_path)
        face_locations = face_recognition.face_locations(face_image_np, model=self.model)
        face_landmarks = face_recognition.face_landmarks(face_image_np, face_locations)
        self._face_img = Image.fromarray(face_image_np)
        self._mask_img = Image.open(self.mask_path)

        found_face = False
        for face_landmark in face_landmarks:
            # check whether facial features meet requirement
            skip = False
            for facial_feature in self.KEY_FACIAL_FEATURES:
                if facial_feature not in face_landmark:
                    skip = True
                    break
            if skip:
                continue

            # mask face
            found_face = True
            self._mask_face(face_landmark)


        if found_face:
            # align
            src_faces = []
            src_face_num = 0
            with_mask_face = np.asarray(self._face_img)
            src_faces.append(with_mask_face)
            faces_aligned =src_faces
            face_num = 0
            for faces in faces_aligned:
                face_num = face_num + 1
                faces = cv2.cvtColor(faces, cv2.COLOR_RGBA2BGR)
                size = (int(160), int(160))
                faces_after_resize = cv2.resize(faces, size, interpolation=cv2.INTER_AREA)
                print("save:",self.save_path)
                # cv2.imwrite(self.save_path, faces_after_resize)
                cv2.imencode('.png', faces_after_resize)[1].tofile(self.save_path)
        else:
            #在这里记录没有裁的图片
            print('Found no face.'+self.save_path)

    def _mask_face(self, face_landmark: dict):
        nose_bridge = face_landmark['nose_bridge']
        nose_point = nose_bridge[len(nose_bridge) * 1 // 4]
        nose_v = np.array(nose_point)

        chin = face_landmark['chin']
        chin_len = len(chin)
        chin_bottom_point = chin[chin_len // 2]
        chin_bottom_v = np.array(chin_bottom_point)
        chin_left_point = chin[chin_len // 8]
        chin_right_point = chin[chin_len * 7 // 8]

        # split mask and resize
        width = self._mask_img.width
        height = self._mask_img.height
        width_ratio = 1.2
        new_height = int(np.linalg.norm(nose_v - chin_bottom_v))

        # left
        mask_left_img = self._mask_img.crop((0, 0, width // 2, height))
        mask_left_width = self.get_distance_from_point_to_line(chin_left_point, nose_point, chin_bottom_point)
        mask_left_width = int(mask_left_width * width_ratio)
        if mask_left_width == 0:
            mask_left_width = 1
        if new_height == 0 :
            new_height =1
        mask_left_img = mask_left_img.resize((mask_left_width, new_height))

        # right
        mask_right_img = self._mask_img.crop((width // 2, 0, width, height))
        mask_right_width = self.get_distance_from_point_to_line(chin_right_point, nose_point, chin_bottom_point)
        mask_right_width = int(mask_right_width * width_ratio)
        if mask_right_width ==0:
            mask_right_width =1
        if new_height ==0:
            new_height =1
        mask_right_img = mask_right_img.resize((mask_right_width, new_height))

        # merge mask
        size = (mask_left_img.width + mask_right_img.width, new_height)
        mask_img = Image.new('RGBA', size)
        mask_img.paste(mask_left_img, (0, 0), mask_left_img)
        mask_img.paste(mask_right_img, (mask_left_img.width, 0), mask_right_img)

        # rotate mask
        angle = np.arctan2(chin_bottom_point[1] - nose_point[1], chin_bottom_point[0] - nose_point[0])
        rotated_mask_img = mask_img.rotate(angle, expand=True)

        # calculate mask location
        center_x = (nose_point[0] + chin_bottom_point[0]) // 2
        center_y = (nose_point[1] + chin_bottom_point[1]) // 2

        offset = mask_img.width // 2 - mask_left_img.width
        radian = angle * np.pi / 180
        box_x = center_x + int(offset * np.cos(radian)) - rotated_mask_img.width // 2
        box_y = center_y + int(offset * np.sin(radian)) - rotated_mask_img.height // 2

        # add mask
        self._face_img.paste(mask_img, (box_x, box_y), mask_img)

    def _save(self):
        path_splits = os.path.splitext(self.face_path)
        new_face_path = path_splits[0] + '-with-mask' + path_splits[1]
        self._face_img.save(new_face_path)
        print(f'Save to {new_face_path}')

    @staticmethod
    def get_distance_from_point_to_line(point, line_point1, line_point2):
        distance = np.abs((line_point2[1] - line_point1[1]) * point[0] +
                          (line_point1[0] - line_point2[0]) * point[1] +
                          (line_point2[0] - line_point1[0]) * line_point1[1] +
                          (line_point1[1] - line_point2[1]) * line_point1[0]) / \
                   np.sqrt((line_point2[1] - line_point1[1]) * (line_point2[1] - line_point1[1]) +
                           (line_point1[0] - line_point2[0]) * (line_point1[0] - line_point2[0]))
        return int(distance)



# if __name__ == '__main__':
#     images_160Dir = "D:/Py_projects/final_facenet/data/images_160"
#     maskedImagesDir = "D:/Py_projects/final_facenet/data/images_masked"
#     wear_mask(images_160Dir, maskedImagesDir)
