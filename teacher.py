"""
description = 0表示未签到 1表示已签到 2表示请假
"""
import sys
import time
import datetime as dt
from threading import Thread
import cv2
import qimage2ndarray

from PyQt5.QtCore import pyqtSignal, Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QMainWindow, QApplication, QMessageBox, QTableWidgetItem

from UI.teacher import Ui_Teacher

from contributed import face
from globals import connect_to_sql, prepare_data


class TeacherWindow(QMainWindow, Ui_Teacher):
    # 信号
    check_signal = pyqtSignal(str)

    # 关闭窗口的时候将所有学生的状态重新赋值未 未签到
    def closeEvent(self, Event):
        db, cursor = connect_to_sql()
        update_sql = "update students set description = 0 where description != 0"
        cursor.execute(update_sql)
        db.commit()
        db.close()
        cursor.close()

    def __init__(self, teacher_name):
        super(TeacherWindow, self).__init__()
        self.setupUi(self)

        # 设置提示字符
        self.lineEdit_supplement.setPlaceholderText("请输入学号")
        self.lineEdit_leave.setPlaceholderText("请输入学号")
        # 初始化教师名字标签
        self.teacher_name = teacher_name
        self.show_teacher_name()
        # 摄像头区域的宽高
        self.width = 800
        self.height = 700
        # 摄像头线程结束标志
        self.is_camera_not_opened = False
        # # 加载识别对象
        # self.face_recognition = face.Recognition()

        self.call_back_functions()
        self.init_classes()

    def show_teacher_name(self):
        hour = dt.datetime.now().hour
        if 6 <= hour < 10:
            self.label_teacher_name.setText(f'早上好, {self.teacher_name} 老师！')
        elif 10 <= hour < 14:
            self.label_teacher_name.setText(f'中午好, {self.teacher_name} 老师！')
        elif 14 <= hour < 18:
            self.label_teacher_name.setText(f'下午好, {self.teacher_name} 老师！')
        else:
            self.label_teacher_name.setText(f'晚上好, {self.teacher_name} 老师！')

    def call_back_functions(self):
        # 打开摄像头
        self.bt_open_camera.clicked.connect(self.open_camera)
        # 重新写配置按钮
        self.rewrite_infos_btn.clicked.connect(self.rewrite_infos)
        # 设置查询班级人数按键的连接函数
        self.bt_check.clicked.connect(self.check_nums)
        # 设置请假登记按键
        self.bt_leave.clicked.connect(self.set_leave_description)
        # 设置漏签补签按键的连接函数
        self.bt_supplement.clicked.connect(self.set_supplement_description)
        # 设置查看结果 显示未到按键的连接函数
        self.bt_view.clicked.connect(self.show_absence)

    def init_classes(self):
        db, cursor = connect_to_sql()
        search_sql = "select class_name from classes"
        cursor.execute(search_sql)
        classes = cursor.fetchall()
        # 将从数据库中获取的班级添加到下拉菜单中
        for item in classes:
            self.classes_comboBox.addItem(item[0])
        db.close()
        cursor.close()

    # 打开摄像头开始考勤， 只要摄像头打开就进入考勤状态，上面的设置都不能写，
    """
    这里我的想法是，不按照班级签到，而是按照上课地点和课程签到，因为一节课可能好几个班同时上
    如果按照班级分类签到的话有几个班就得重新设置几次，太麻烦
    所以直接签到，扫到哪个人然后从数据库中扒出来他的信息，然后设置签到状态
    但是这样就存在一个问题，如何判断这个人该不该来上这节课。
    也就是说需要从教务处获取到这个时间，这个地点需要上什么课，哪些班需要来上，这些班都有哪些人
    有了这些信息才能在判断这个人是否需要签到，签到完成后才能判断哪些人没来
    """
    def open_camera(self):
        # class_name = self.classes_comboBox.currentText()
        location = self.lineEdit_location.text()
        course = self.lineEdit_course.text()
        if location and course:
            try:
                db, cursor = connect_to_sql()
                # 按照 日期-老师-课程名称-地点 的形式创建表 表名中不能有'-'
                today = str(dt.date.today()).replace('-', '_')
                table_name = f"{today}_{self.teacher_name}_{course}_{location}"
                create_sql = """create table %s(
                                 student_id int(10) not null primary key,
                                 student_name varchar(25),
                                 student_class varchar(25),
                                 check_in_time varchar(25)
                                 )
                                 """ % table_name
                cursor.execute(create_sql)
            # except Exception as e:
            #     print(str(e))
            finally:
                db.commit()
                db.close()
                cursor.close()
            self.set_widgets_disabled()
            self.start_camera(table_name)
        else:
            QMessageBox.warning(self, "Warning", "请输入完整的上课地点和课程！", QMessageBox.Ok)

    def set_widgets_disabled(self):
        self.classes_comboBox.setEnabled(False)
        self.lineEdit_location.setEnabled(False)
        self.lineEdit_course.setEnabled(False)
        self.bt_open_camera.setEnabled(False)

    def start_camera(self, table_name):
        try:
            self.camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            self.textBrowser_log.append('正在启动，请稍后...')
        except Exception as e:
            self.people_textBrowser.clear()
            self.people_textBrowser.append(str(e))
        finally:
            # 创建子线程，识别人脸
            self.recognition_thread = Thread(target=self.face_recognition_func, args=(table_name,), daemon=True)
            self.check_signal.connect(self.check_signal_call_back_func)
            self.recognition_thread.start()

    def face_recognition_func(self, table_name):
        frame_interval = 10  # Number of frames after which to run face detection
        frame_count = 0
        face_recognition = face.Recognition()
        db, cursor = connect_to_sql()

        while True:
            success, frame = self.camera.read()
            if success:
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                qimg = qimage2ndarray.array2qimage(img)
                self.label_camera.setPixmap(QPixmap(qimg).scaled(self.width, self.height))
                # self.img = img
                # 识别
                if (frame_count % frame_interval) == 0:
                    # 获取当前时间
                    current_time = (str(dt.datetime.today()).split('.')[0]).split(' ')[1]
                    faces = face_recognition.identify(frame)
                    self.find_student(faces, current_time, db, cursor, table_name, frame)
                frame_count += 1
            else:
                pass
            # 关闭摄像头
            if self.is_camera_not_opened:
                break

    def find_student(self, faces, current_time, db, cursor, table_name, frame):
        if faces is not None:
            for face in faces:
                # face_bb = face.bounding_box.astype(int)
                # cv2.rectangle(frame,
                #               (face_bb[0], face_bb[1]), (face_bb[2], face_bb[3]),
                #               (0, 255, 0), 2)
                if face.name is not None:
                    stu_id = face.name.split('_')[0]
                    search_sql = "select ID, student_name, student_class from students where ID = '%s'" % stu_id
                    cursor.execute(search_sql)
                    result = cursor.fetchall()
                    print(result)
                    # 如果找匹配成功,则往新建的表中插入数据，同时修改students表中的description
                    if result:
                        data = prepare_data(result[0], current_time)
                        insert_sql = f"replace into {table_name}(student_id,student_name,student_class,check_in_time) values(%s,%s,%s,%s)"
                        cursor.execute(insert_sql, data)
                        update_sql = "update students set description = 1 where student_name = '%s'" %(face.name)
                        cursor.execute(update_sql)
                        self.check_signal.emit(f'{face.name}  签到成功！ 请离开摄像头范围。')
                        time.sleep(0.2)
                        db.commit()
                else:
                    self.check_signal.emit('未录入学生，请先录入学生信息!')

    def check_signal_call_back_func(self, msg):
        self.textBrowser_log.append(msg)

    def rewrite_infos(self):
        self.is_camera_not_opened = True
        self.set_widgets_enabled()

    def set_widgets_enabled(self):
        self.classes_comboBox.setEnabled(True)
        self.lineEdit_location.setEnabled(True)
        self.lineEdit_course.setEnabled(True)
        self.bt_open_camera.setEnabled(True)

    # 查询签到人数按钮
    def check_nums(self):
        # 记录已签到的人
        checked_students = []
        class_checked = self.classes_comboBox.currentText()
        db, cursor = connect_to_sql()
        search_sql = "select * from students where student_class = '%s'" % class_checked
        cursor.execute(search_sql)
        result = cursor.fetchall()
        all_student_num = len(result)
        for item in result:
            if item[3] != 0:
                checked_students.append(item)
        checked_student_num = len(checked_students)
        self.lcd_1.display(str(all_student_num))
        self.lcd_2.display(str(checked_student_num))
        db.close()
        cursor.close()

    # 请假登记按钮
    def set_leave_description(self):
        leave_student_id = self.lineEdit_leave.text()
        if leave_student_id:
            self.execution(leave_student_id, 2)
            self.lineEdit_leave.clear()
        else:
            QMessageBox.warning(self, "Warning", "请输入请假的学生学号！", QMessageBox.Ok)

    # 漏签补签按钮
    def set_supplement_description(self):
        supplement_student_id = self.lineEdit_supplement.text()
        if supplement_student_id:
            self.execution(supplement_student_id, 1)
            self.lineEdit_supplement.clear()
        else:
            QMessageBox.warning(self, "Warning", "请输入需要补签的学生学号！", QMessageBox.Ok)

    # 请假登记按钮
    def execution(self, id, status):
        db, cursor = connect_to_sql()
        search_sql = "select * from students where ID = '%s'" % id
        cursor.execute(search_sql)
        result = cursor.fetchall()
        if result:
            update_sql = "update students set description = %s where ID = '%s'" % (status, id)
            cursor.execute(update_sql)
            db.commit()
            db.close()
            cursor.close()
            QMessageBox.information(self, "Information", f"学号：{result[0][0]}, 姓名：{result[0][1]} 状态已变更,请勿重复操作!", QMessageBox.Ok)
        else:
            QMessageBox.warning(self, "Warning", "输入的学号不存在！", QMessageBox.Ok)

    # 查看未签到的人
    def show_absence(self):
        db, cursor = connect_to_sql()
        search_sql = "select student_name, student_class from students where description = 0"
        cursor.execute(search_sql)
        data = cursor.fetchall()
        row = cursor.rowcount  # 取得记录个数，用于设置表格的行数
        vol = len(data[0])  # 取得字段数，用于设置表格的列数
        self.tableWidget_absent_students.setRowCount(row)
        self.tableWidget_absent_students.setColumnCount(vol)

        for row, row_value in enumerate(data):
            for col, col_value in enumerate(row_value):
                item = QTableWidgetItem(str(col_value))
                self.tableWidget_absent_students.setItem(row, col, item)
        db.close()
        cursor.close()

    def handle_click(self):
        if not self.isVisible():
            self.show()


# if __name__ == '__main__':
#     app = QApplication(sys.argv)
#     teacher_win = TeacherWindow('Teacher')
#     teacher_win.show()
#     sys.exit(app.exec_())
