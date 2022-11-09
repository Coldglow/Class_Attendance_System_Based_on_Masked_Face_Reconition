import sys

from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QTabWidget, QMessageBox, QApplication, QTableWidgetItem

from UI.admin import Ui_Admin

import imageio
import qimage2ndarray
import cv2
import os
import time
import shutil
import threading

from globals import connect_to_sql, prepare_data
from src.align.align_dataset_mtcnn import process_all_160
from src.align.align_dataset_mtcnn import process_this_160
from mixImages import mix_images
from src.classifier import process_classifier
from wearmask import cli

savedImagesDir = "data/images"
images_160Dir = "data/images_160"
maskedImagesDir = "data/images_masked"
mixedImagesDir = "data/images_mixed"
pbFileDir = "model/20220227-201123/lite_s.pb"
pklFileDir = "model/peoples.pkl"

IMAGE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mask_images')
cloth_PATH = os.path.join(IMAGE_DIR, 'cloth.png')
surgical_white_PATH = os.path.join(IMAGE_DIR, 'surgical.png')
surgical_blue_IMAGE_PATH = os.path.join(IMAGE_DIR, 'surgical_blue.png')
surgical_green_IMAGE_PATH = os.path.join(IMAGE_DIR, 'surgical_green.png')
KN95_IMAGE_PATH = os.path.join(IMAGE_DIR, 'KN95.png')


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


# 佩戴口罩子线程定义
class Wearthread(QThread):
    #  通过类成员对象定义信号对象
    wear_signal = pyqtSignal(str, bool)

    def __init__(self, mask_path):  # 默认为随机选择口罩
        super(Wearthread, self).__init__()
        self.mask_path = mask_path

    def run(self):
        filelist = os.listdir(images_160Dir)  # 统计文件总数
        total_task_number = len(filelist)
        task_number = 0  # 表示正在执行第几个任务
        flag = False
        # 遍历文件佩戴口罩并且重命名
        for root, dirs, files in os.walk(images_160Dir, topdown=False):

            task_number += 1
            if task_number < total_task_number:
                msg = f"正在处理第{task_number}/{total_task_number}个任务..."
                self.wear_signal.emit(msg, flag)  # 发送实时任务进度和总任务进度
            elif task_number == total_task_number:
                msg = "图片处理完成！\n正在更新人脸图像..."
                flag = True
                self.wear_signal.emit(msg, flag)  # 发送实时任务进度和总任务进度

            i = 0  # i表示图片名字末尾的数字 代表一个人的第几张图
            for name in files:  # name表示图片本身
                new_root = root.replace(images_160Dir, maskedImagesDir)
                if not os.path.exists(new_root):  # 如果保存图片的途径不存在则创建
                    os.makedirs(new_root)
                # deal
                imgpath = os.path.join(root, name)  # imgpath表示旧图片路径
                save_imgpath = os.path.join(new_root, name)  # save_imgpath表示新的戴上口罩后的图片的路径

                n = name.split("_0")[0]  # 取出用户的名字
                check_exist = new_root + '/' + str(n) + '_1' + format(str(i + 1), '0>3s') + '.png'  # 重命名后的第一张图片

                if os.path.exists(check_exist):  # 如果路径存在，则表示戴了口罩且已经重命名，继续检查下一个
                    continue
                # if os.path.exists(check_exist):  # 如果路径存在，则表示戴了口罩且已经重命名，那么需要删除该图片，重新佩戴
                #     os.remove(check_exist)
                # 若上述路径不存在，则表示未带口罩需要带口罩 执行cli函数
                cli(imgpath, save_imgpath, self.mask_path)
                # 如果佩戴成功，则重命名图片
                if os.path.exists(save_imgpath):
                    while os.path.exists(check_exist):
                        i += 1
                    # src是旧的名字  dst是新名字
                    src = new_root + '/' + name
                    dst = new_root + '/' + str(n) + '_1' + format(str(i + 1), '0>3s') + '.png'
                    os.rename(src, dst)
                    i = i + 1


# 训练模型子线程定义，在这之前
class TrainThread(QThread):
    #  通过类成员对象定义信号对象
    train_signal = pyqtSignal(int, int)

    def __init__(self):
        super(TrainThread, self).__init__()

    def run(self):
        # 合并图片
        mix_images(images_160Dir, maskedImagesDir, mixedImagesDir)

        # 训练模型
        data = process_classifier(pbFileDir, mixedImagesDir, pklFileDir)
        self.train_signal.emit(data.get('classes'), data.get('images'))


class AdminWindow(QTabWidget, Ui_Admin):

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
        super(AdminWindow, self).__init__(parent)
        self.setupUi(self)
        self.CallBackFunctions()
        # 显示下拉菜单中的班级名称
        self.init_classes()

        # 相机显示长宽
        self.width = self.cameraShow.width()
        self.height = self.cameraShow.height()

        # 点击检测会保存9张图片
        self.saved_pic_num = 9
        # False表示只会对在界面拍照得到的图片对齐 True表示会对文件夹中所有人的图片对齐 只需修改这里即可
        self.has_class_directories = False

    def CallBackFunctions(self):
        self.openCamBtn.clicked.connect(self.StartCamera)
        # 点击检测按钮会录入10张图片并且分割图片，修改名称，并将分割好的图片显示在右侧
        self.verifactionBtn.clicked.connect(self.save_pic)
        # 点击删除人员，会把对应的人从文件夹中删除
        self.delPeopleBtn.clicked.connect(self.delMember)
        # 更新人员按钮，执行佩戴口罩的操作，然后重新训练分类器
        self.update_SVM_btn.clicked.connect(self.update_pepele)
        # 核验身份，验证数据库中的学生数量是否和人脸文件中身份数量相同
        self.check_variation_btn.clicked.connect(self.verification)

        ###### 班级和人员管理界面连接函数 #######
        # 添加班级按钮
        self.add_class_btn.clicked.connect(self.add_class_func)
        # 删除班级
        self.del_class_btn.clicked.connect(self.del_class_func)
        # # 添加学生
        self.add_student_btn.clicked.connect(self.add_student_func)
        # # 删除学生
        self.del_student_btn.clicked.connect(self.del_student_func)
        # # 查看所有学生
        self.check_all_students.clicked.connect(self.check_all_students_func)
        # # 查看所有班级
        self.check_all_classes.clicked.connect(self.check_all_classes_func)

    def update_pepele(self):
        click = QMessageBox.information(self, '消息',
                            "您确定更新人员么？", QMessageBox.Ok)
        if click == QMessageBox.Ok:
            # 创建戴口罩线程和训练的线程
            self.wear_random_mask_thread = Wearthread(None)
            self.train_thread = TrainThread()
            # 连接信号
            self.wear_random_mask_thread.wear_signal.connect(self.wear_mask_call_back_func)
            self.train_thread.train_signal.connect(self.train_svm_call_back_func)
            # 先启动带口罩的线程，在带口罩的线程中信号的返回函数中启动训练线程
            self.people_textBrowser.append('模型启动中...')
            self.wear_random_mask_thread.start()

    def wear_mask_call_back_func(self, msg, flag):
        self.people_textBrowser.append(msg)
        if flag:
            self.train_thread.start()  # 开始更新SVM

    def train_svm_call_back_func(self, classes, images, ):
        self.people_textBrowser.append("更新完成！\n身份数量：" + str(classes) + "\n图片数量：" + str(images))

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

    # 删除人员
    def delMember(self):
        stu_id = self.id_Input.text()
        stu_name = self.nameInput.text()
        flag = True
        if stu_id and stu_name:
            for value in [savedImagesDir, images_160Dir, maskedImagesDir, mixedImagesDir]:
                class_file_name = value + "/" + stu_id + "_" + stu_name
                if os.path.exists(class_file_name):
                    shutil.rmtree(class_file_name)
                else:
                    QMessageBox.warning(self, "Warning", f"学号：{stu_id}，姓名：{stu_name}不存在，请重新输入。", QMessageBox.Ok)
                    flag = False
                    break
            if flag:
                QMessageBox.information(self, "Information", f"身份 {stu_name} 删除成功，请勿重复操作！", QMessageBox.Ok)
        else:
            QMessageBox.warning(self, "Warning", "请输入要删除的学生学号和姓名", QMessageBox.Ok)

    # 检查数据库和本地文件身份信息是否相同
    def verification(self):
        """
        database_set: 从数据库中获取的{(id， name)}集合
        local_classes_set：本地文件中身份的集合
        :return:
        """
        database_set = self.get_database_set()
        local_classes_set = self.get_local_classes_set()
        # 两者相减
        database_sub_local = database_set - local_classes_set
        local_sub_database = local_classes_set - database_set

        if database_sub_local or local_sub_database:
            self.people_textBrowser.append("人脸信息和学生信息不匹配！")
            if database_sub_local:
                self.people_textBrowser.append("存在班级信息但缺失人脸图片的学生：")
                for item in database_sub_local:
                    self.people_textBrowser.append(str(item))
                self.people_textBrowser.append("以上学生请尽快录入人脸信息！")

            if local_sub_database:
                self.people_textBrowser.append("存在人脸信息但缺失班级信息的学生：")
                for item in local_sub_database:
                    self.people_textBrowser.append(str(item))
                self.people_textBrowser.append("以上学生请尽快录入班级信息！")

    def get_database_set(self):
        db, cursor = connect_to_sql()
        search_sql = "select ID, student_name from students"
        cursor.execute(search_sql)
        database_set = set(cursor.fetchall())
        db.close()
        cursor.close()

        return database_set

    def get_local_classes_set(self):
        classes = os.listdir(mixedImagesDir)
        all_local_classes_set = set()
        for item in classes:
            id = int(item.split("_")[0])
            name = item.split("_")[1]
            single_class_set = (id, name)
            all_local_classes_set.add(single_class_set)

        return all_local_classes_set

    ####################### 下面是班级学生管理界面的操作  ###################
    # 初始化下拉菜单
    def init_classes(self):
        db, cursor = connect_to_sql()
        search_sql = "select class_name from classes"
        cursor.execute(search_sql)
        classes = cursor.fetchall()
        # 将从数据库中获取的班级添加到下拉菜单中
        for item in classes:
            self.add_student_class_combox.addItem(item[0])
        db.close()
        cursor.close()

    # 添加班级
    def add_class_func(self):
        class_name = self.add_class_name_input.text()
        class_number = int(self.add_student_number_input.text())
        if not class_name or not class_number:
            QMessageBox.warning(self, '消息',"请输入完整班级信息")
        else:
            # 连接数据库
            try:
                data = prepare_data(class_name, class_number)
                db, cursor = connect_to_sql()
                insert_sql = "replace into classes(class_name, class_num) values(%s, %s)"
                cursor.execute(insert_sql, data)
                QMessageBox.information(self, "Information", "添加成功，请勿重复操作！", QMessageBox.Ok)
            except Exception as e:
                db.rollback()
                print(str(e))
                QMessageBox.warning(self, "Warning", str(e), QMessageBox.Ok)
            finally:
                db.commit()
                db.close()
                cursor.close()
        self.add_class_name_input.clear()
        self.add_student_number_input.clear()
        self.add_student_class_combox.addItem(class_name)  # 将添加的班级加入到下拉菜单中

    # 删除班级
    def del_class_func(self):
        class_name = self.del_class_name_input.text()
        if not class_name:
            QMessageBox.warning(self, '消息',
                                "请输入要删除的班级名称")
        else:
            try:
                db, cursor = connect_to_sql()
                # select默认将查询的结果放在一个元组中，如果没有查到，则元组为空
                search_sql = "select * from classes where class_name = '%s'" % (class_name)
                cursor.execute(search_sql)
                # print(cursor.fetchall())
                if cursor.fetchall():  # 如果查到，则删除
                    del_sql = "delete from classes where class_name = '%s'" % (class_name)
                    cursor.execute(del_sql)
                    QMessageBox.information(self, "Information", f"{class_name}删除成功，请勿重复操作！", QMessageBox.Ok)
                else:
                    QMessageBox.warning(self, "Warning", f"班级{class_name}不存在！", QMessageBox.Ok)
            except Exception as e:
                db.rollback()
                print(str(e))
                QMessageBox.warning(self, "Warning", str(e), QMessageBox.Ok)
            finally:
                db.commit()
                db.close()
                cursor.close()
        self.del_class_name_input.clear()
        # 因为没有删除item的函数 所以先清除再重新初始化
        self.add_student_class_combox.clear()
        self.init_classes()

    # 添加学生
    def add_student_func(self):
        student_ID = int(self.add_student_ID_input.text())
        student_name = self.add_student_name_input.text()
        class_name = self.add_student_class_combox.currentText()
        if not class_name or not student_ID or not student_name:
            QMessageBox.warning(self, '消息',
                                "请输入完整学生信息")
        else:
            # 连接数据库
            try:
                data = prepare_data(student_ID, student_name, class_name)
                db, cursor = connect_to_sql()
                insert_sql = "replace into students(ID, student_name, student_class) values(%s, %s, %s)"
                cursor.execute(insert_sql, data)
                QMessageBox.information(self, "Information", f"学生{student_name}添加成功，请勿重复操作！", QMessageBox.Ok)
            except Exception as e:
                db.rollback()
                print(str(e))
                QMessageBox.warning(self, "Warning", str(e), QMessageBox.Ok)
            finally:
                db.commit()
                db.close()
                cursor.close()
        self.add_student_ID_input.clear()
        self.add_student_name_input.clear()
        self.add_class_name_input.clear()

    # 删除学生
    def del_student_func(self):
        student_id = int(self.del_student_ID_input.text())
        if not student_id:
            QMessageBox.warning(self, '消息',
                                "请输入要删除学生的学号")
        else:
            try:
                db, cursor = connect_to_sql()
                # select默认将查询的结果放在一个元组中，如果没有查到，则元组为空
                search_sql = "select * from students where ID = '%s'" % (student_id)
                cursor.execute(search_sql)
                student_info = cursor.fetchall()
                if student_info:  # 如果查到，则删除
                    del_sql = "delete from classes where class_name = '%s'" % (student_id)
                    cursor.execute(del_sql)
                    QMessageBox.information(self, "Information", f"学号：{student_id}. 姓名：{student_info[0][1]}. 删除成功，请勿重复操作！",
                                        QMessageBox.Ok)
                else:
                    QMessageBox.warning(self, "Warning", f"学号：{student_id}不存在！", QMessageBox.Ok)
            except Exception as e:
                db.rollback()
                print(str(e))
                QMessageBox.warning(self, "Warning", str(e), QMessageBox.Ok)
            finally:
                db.commit()
                db.close()
                cursor.close()
        self.del_class_name_input.clear()

    def check_all_students_func(self):
        db, cursor = connect_to_sql()
        search_students_sql = "select ID, student_name, student_class from students"
        cursor.execute(search_students_sql)
        data = cursor.fetchall()
        # 必须先设置行数和列数。否则显示不出来
        row = cursor.rowcount  # 取得记录个数，用于设置表格的行数
        vol = len(data[0])  # 取得字段数，用于设置表格的列数
        self.students_table_widget.setColumnCount(vol)
        self.students_table_widget.setRowCount(row)

        for row, row_value in enumerate(data):
            for col, col_value in enumerate(row_value):
                item = QTableWidgetItem(str(col_value))
                item.setTextAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
                self.students_table_widget.setItem(row, col, item)
        db.close()
        cursor.close()

    def check_all_classes_func(self):
        db, cursor = connect_to_sql()
        search_students_sql = "select class_name, class_num from classes"
        cursor.execute(search_students_sql)
        data = cursor.fetchall()
        # 必须先设置行数和列数。否则显示不出来
        row = cursor.rowcount  # 取得记录个数，用于设置表格的行数
        vol = len(data[0])  # 取得字段数，用于设置表格的列数
        self.classes_table_widget.setRowCount(row)
        self.classes_table_widget.setColumnCount(vol)

        for row, row_value in enumerate(data):
            for col, col_value in enumerate(row_value):
                item = QTableWidgetItem(str(col_value))
                item.setTextAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
                self.classes_table_widget.setItem(row, col, item)
        db.close()
        cursor.close()

    ##################### 全局操作 ###################################

    def handle_click(self):
        if not self.isVisible():
            self.show()


# if __name__ == '__main__':
#     app = QApplication(sys.argv)
#     main_window = AdminWindow()
#     main_window.show()
#     sys.exit(app.exec_())
