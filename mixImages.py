import os
import shutil


def CopyFile(filepath, newPath):
    # 获取当前路径下的文件名，返回List
    images = os.listdir(filepath)
    for image in images:
        # 将文件加入到当前文件路径后面
        image_dir = filepath + '/' + image
        # 如果是文件
        if os.path.isfile(image_dir):
            print(image_dir)
            image_new_dir = newPath + '/' + image
            # copyfile函数两个必须为文件，不能是目录，
            shutil.copyfile(image_dir, image_new_dir)
        # 如果不是文件，递归这个文件夹的路径
        else:
            print(image_dir + "is not a file")
            CopyFile(image_dir, newPath)


def mix_images(images_160_dir, images_masked_dir, images_mixed_dir):
    # 若不存在目标文件，则创建
    if not os.path.exists(images_mixed_dir):
        # 如果不存在则创建目录
        os.makedirs(images_mixed_dir)

    #复制图片
    for classes in os.listdir(images_160_dir):
        if not os.path.exists(images_mixed_dir + '/' + classes):
            os.makedirs(images_mixed_dir + '/' + classes)
            CopyFile(images_160_dir + '/' + classes, images_mixed_dir + "/" + classes)
            CopyFile(images_masked_dir + '/' + classes, images_mixed_dir + "/" + classes)


if __name__ == '__main__':
    images_160Dir = "D:/Graduation_Project/FaceRecognitionAttendanceSystem/data/images_160"
    maskedImagesDir = "D:/Graduation_Project/FaceRecognitionAttendanceSystem/data/images_masked"
    mixedImagesDir = "D:/Graduation_Project/FaceRecognitionAttendanceSystem/data/images_mixed"
    mix_images(images_160Dir, maskedImagesDir, mixedImagesDir)

# D:\Graduation_Project\FaceRecognitionAttendanceSystem\data\images_mixed