from PyQt5.QtWidgets import QApplication, QWidget, QMessageBox
from UI.login import Ui_Login

from admin import AdminWindow
from teacher import TeacherWindow
from student import StudentWindow
from globals import connect_to_sql
import sys


class MainWindow(QWidget, Ui_Login):

    def __init__(self):
        super(MainWindow, self).__init__()
        self.setupUi(self)
        self.call_back_functions()
        self.switch = None

    def call_back_functions(self):
        self.login_btn.clicked.connect(self.user_login)
        self.admin_btn.toggled.connect(lambda: self.btn_state(self.admin_btn))
        self.teacher_btn.toggled.connect(lambda: self.btn_state(self.teacher_btn))
        self.stu_btn.toggled.connect(lambda: self.btn_state(self.stu_btn))

    def user_login(self):
        name = self.user_name_input.text()
        pwd = self.pwd_input.text()
        sql = "select * from users where userName='%s' and pwd='%s'" % (name, pwd)
        if name and pwd:
            try:
                db, cursor = connect_to_sql()
                cursor.execute(sql)
                result = cursor.fetchall()
                num = len(result)
                if num == 1:
                    main_window.close()
                    if self.switch == 0:
                        self.admin_window = AdminWindow()
                        self.admin_window.handle_click()
                    elif self.switch == 1:
                        teacher_name = result[0][2]
                        self.teacher_window = TeacherWindow(teacher_name)
                        self.teacher_window.handle_click()
                    elif self.switch == 2:
                        self.stu_window = StudentWindow()
                        self.stu_window.handle_click()
                    else:
                        QMessageBox.warning(self, '消息',
                                            "请选择用户类型")
                else:
                    QMessageBox.warning(self, '消息',
                                        "用户名或密码错误")
            except Exception as e:
                print(str(e))
            finally:
                cursor.close()
                db.close()
        else:
            QMessageBox.warning(self, "Warning", "请输入用户名或密码", QMessageBox.Ok)

    def btn_state(self, btn):
        if btn.isChecked() and btn.text() == '管理员':
            self.switch = 0
        elif btn.isChecked() and btn.text() == '教师':
            self.switch = 1
        elif btn.isChecked() and btn.text() == '学生':
            self.switch = 2


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())
