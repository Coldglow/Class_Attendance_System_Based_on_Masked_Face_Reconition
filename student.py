from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QTabWidget, QMessageBox, QApplication, QTableWidgetItem

from UI.student import Ui_Student

import sys
import imageio
import qimage2ndarray
import cv2
import os
import time
import threading

from src.align.align_dataset_mtcnn import process_all_160
from src.align.align_dataset_mtcnn import process_this_160


savedImagesDir = "data/images"
images_160Dir = "data/images_160"
maskedImagesDir = "data/images_masked"
mixedImagesDir = "data/images_mixed"


# 图片对齐操作线程
class AlignThread(QThread):
    input_signal = pyqtSignal(str)
    align_signal = pyqtSignal(str)

    def __init__(self, member):  # 参数member为MemberManage对象，即表示主界面
        super(AlignThread, self).__init__()
        self.daemon = True  # 设置守护线程
        self.member = member
        self.member_name = self.member.nameInput.text()
        self.class_file_name = self.member.id_Input.text() + "_" + self.member.nameInput.text()

    def run(self):
        # 写入图片
        for i in range(self.member.saved_pic_num):
            imageio.imwrite(
                savedImagesDir + "/" + self.class_file_name + '/' + self.member_name + '_000' + str(i + 1) + '.png',
                self.member.img)
            time.sleep(0.1)
        self.input_signal.emit(f'{self.member_name}录入成功!\n{self.member_name}对齐中...')

        # 对齐图片
        if self.member.has_class_directories:
            process_all_160(savedImagesDir, images_160Dir)
        else:
            input_class = savedImagesDir + '/' + self.class_file_name
            output_class = images_160Dir + '/' + self.class_file_name
            process_this_160(input_class, output_class)
        self.align_signal.emit(f'{self.member_name}对齐成功!')


class StudentWindow(QTabWidget, Ui_Student):

    # 重写closeEvent方法，在关闭窗口时执行
    def closeEvent(self, event):
        try:
            self.camera.release()  # 释放资源
            self.is_camera_not_opened = True  # 退出摄像头子线程
            # self.camera_thread.join()
            self.openCamBtn.setEnabled(True)  # 设置打开摄像机按钮可以可以使用
            self.cameraShow.clear()  # 摄像头显示区域清空
            self.nameInput.clear()
            self.img160Show.clear()
        except Exception as e:
            print(str(e))

    def __init__(self, parent=None):
        # super()构造器方法返回父级的对象。__init__()方法是构造器的一个方法。
        super(StudentWindow, self).__init__(parent)
        self.setupUi(self)
        self.CallBackFunctions()

        # 相机显示长宽
        self.width = self.cameraShow.width()
        self.height = self.cameraShow.height()

        # 点击检测会保存9张图片
        self.saved_pic_num = 9
        # False表示只会对在界面拍照得到的图片对齐 True表示会对文件夹中所有人的图片对齐 只需修改这里即可
        self.has_class_directories = False

    def CallBackFunctions(self):
        self.openCamBtn.clicked.connect(self.StartCamera)
        # 点击检测按钮会录入9张图片并且分割图片，修改名称，并将分割好的图片显示在右侧
        self.verifactionBtn.clicked.connect(self.save_pic)

    def save_pic(self):
        stu_id = self.id_Input.text()
        stu_name = self.nameInput.text()
        # 若输入姓名则保存，否则不保存
        if stu_id and stu_name:
            self.verifactionBtn.setEnabled(False)
            file = savedImagesDir + "/" + stu_id + "_" + stu_name
            if os.path.exists(file):
                pass
            else:
                os.mkdir(file)
            self.people_textBrowser.append(f'{stu_name}录入中...')
            # inwrite（保存的文件名，保存的图片） 写入9张
            self.create_thread_write_pic()
        else:
            pass

    # 开始子线程
    def create_thread_write_pic(self):
        self.aligin_threaad = AlignThread(self)  # 创建对其图片线程
        self.aligin_threaad.input_signal.connect(self.input_textbrowser)
        self.aligin_threaad.align_signal.connect(self.align_fun)
        self.aligin_threaad.start()

    def input_textbrowser(self, msg):
        self.people_textBrowser.append(msg)

    def align_fun(self, msg):
        self.people_textBrowser.append(msg)
        class_name = self.id_Input.text() + "_" + self.nameInput.text()
        # 将对齐后的图片展示在label中
        imgName = images_160Dir + "/" + class_name + "/" + self.nameInput.text() + "_0001.png"
        img160 = QPixmap(imgName).scaled(self.img160Show.width(), self.img160Show.height())
        self.img160Show.setPixmap(img160)

        # 继续录入下一位
        self.verifactionBtn.setEnabled(True)

    def StartCamera(self):
        # 摄像头线程结束标志
        self.is_camera_not_opened = False
        try:
            self.camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        except Exception as e:
            self.people_textBrowser.clear()
            self.people_textBrowser.append(str(e))

        self.camera_thread = threading.Thread(target=self.show_frame, daemon=True)  # 摄像机线程
        self.openCamBtn.setEnabled(False)
        self.camera_thread.start()

    def show_frame(self):
        while True:
            success, frame = self.camera.read()
            if success:
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                qimg = qimage2ndarray.array2qimage(img)
                self.cameraShow.setPixmap(QPixmap(qimg).scaled(self.width, self.height))
                self.img = img
            else:
                pass
            # 关闭摄像头
            if self.is_camera_not_opened:
                break

    ##################### 全局操作 ###################################

    def handle_click(self):
        if not self.isVisible():
            self.show()


# if __name__ == '__main__':
#     app = QApplication(sys.argv)
#     main_window = StudentWindow()
#     main_window.show()
#     sys.exit(app.exec_())
