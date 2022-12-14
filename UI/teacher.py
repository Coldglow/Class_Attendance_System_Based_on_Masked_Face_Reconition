# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'teacher.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Teacher(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1136, 924)
        MainWindow.setStyleSheet("QMainWindow {\n"
"    background-color: #ffffff;\n"
"}\n"
"\n"
"QPushButton {\n"
"       color: #212121; \n"
"        border-radius: 10px; \n"
"        border: 1px groove gray;\n"
"        background-color: #F5F7FA\n"
"}\n"
"\n"
" QPushButton:hover {\n"
"        background-color: #d9d9f3 ;\n"
" }")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.horizontalLayout_12 = QtWidgets.QHBoxLayout(self.centralwidget)
        self.horizontalLayout_12.setObjectName("horizontalLayout_12")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout()
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.label_camera = QtWidgets.QLabel(self.centralwidget)
        self.label_camera.setMinimumSize(QtCore.QSize(800, 700))
        font = QtGui.QFont()
        font.setFamily("Mongolian Baiti")
        self.label_camera.setFont(font)
        self.label_camera.setText("")
        self.label_camera.setObjectName("label_camera")
        self.verticalLayout_3.addWidget(self.label_camera)
        self.textBrowser_log = QtWidgets.QTextBrowser(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("新宋体")
        font.setPointSize(15)
        self.textBrowser_log.setFont(font)
        self.textBrowser_log.setObjectName("textBrowser_log")
        self.verticalLayout_3.addWidget(self.textBrowser_log)
        self.horizontalLayout_12.addLayout(self.verticalLayout_3)
        self.line_2 = QtWidgets.QFrame(self.centralwidget)
        self.line_2.setFrameShape(QtWidgets.QFrame.VLine)
        self.line_2.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_2.setObjectName("line_2")
        self.horizontalLayout_12.addWidget(self.line_2)
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.horizontalLayout_9 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_9.setObjectName("horizontalLayout_9")
        self.label_teacher_name = QtWidgets.QLabel(self.centralwidget)
        self.label_teacher_name.setMinimumSize(QtCore.QSize(0, 40))
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(15)
        self.label_teacher_name.setFont(font)
        self.label_teacher_name.setText("")
        self.label_teacher_name.setObjectName("label_teacher_name")
        self.horizontalLayout_9.addWidget(self.label_teacher_name)
        self.verticalLayout_2.addLayout(self.horizontalLayout_9)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label_class = QtWidgets.QLabel(self.centralwidget)
        self.label_class.setMaximumSize(QtCore.QSize(65, 30))
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(10)
        self.label_class.setFont(font)
        self.label_class.setObjectName("label_class")
        self.horizontalLayout.addWidget(self.label_class)
        self.classes_comboBox = QtWidgets.QComboBox(self.centralwidget)
        self.classes_comboBox.setMinimumSize(QtCore.QSize(185, 30))
        font = QtGui.QFont()
        font.setFamily("新宋体")
        font.setPointSize(10)
        self.classes_comboBox.setFont(font)
        self.classes_comboBox.setObjectName("classes_comboBox")
        self.horizontalLayout.addWidget(self.classes_comboBox)
        self.verticalLayout_2.addLayout(self.horizontalLayout)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.label_location = QtWidgets.QLabel(self.centralwidget)
        self.label_location.setMinimumSize(QtCore.QSize(0, 30))
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(10)
        self.label_location.setFont(font)
        self.label_location.setObjectName("label_location")
        self.horizontalLayout_2.addWidget(self.label_location)
        self.lineEdit_location = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_location.setMinimumSize(QtCore.QSize(0, 30))
        font = QtGui.QFont()
        font.setFamily("新宋体")
        font.setPointSize(10)
        self.lineEdit_location.setFont(font)
        self.lineEdit_location.setObjectName("lineEdit_location")
        self.horizontalLayout_2.addWidget(self.lineEdit_location)
        self.verticalLayout_2.addLayout(self.horizontalLayout_2)
        self.horizontalLayout_8 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_8.setObjectName("horizontalLayout_8")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setMinimumSize(QtCore.QSize(0, 30))
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(10)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.horizontalLayout_8.addWidget(self.label)
        self.lineEdit_course = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_course.setMinimumSize(QtCore.QSize(0, 30))
        font = QtGui.QFont()
        font.setFamily("新宋体")
        font.setPointSize(10)
        self.lineEdit_course.setFont(font)
        self.lineEdit_course.setObjectName("lineEdit_course")
        self.horizontalLayout_8.addWidget(self.lineEdit_course)
        self.verticalLayout_2.addLayout(self.horizontalLayout_8)
        self.bt_open_camera = QtWidgets.QPushButton(self.centralwidget)
        self.bt_open_camera.setMinimumSize(QtCore.QSize(0, 50))
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(12)
        self.bt_open_camera.setFont(font)
        self.bt_open_camera.setObjectName("bt_open_camera")
        self.verticalLayout_2.addWidget(self.bt_open_camera)
        self.rewrite_infos_btn = QtWidgets.QPushButton(self.centralwidget)
        self.rewrite_infos_btn.setMinimumSize(QtCore.QSize(0, 50))
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(12)
        self.rewrite_infos_btn.setFont(font)
        self.rewrite_infos_btn.setObjectName("rewrite_infos_btn")
        self.verticalLayout_2.addWidget(self.rewrite_infos_btn)
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.label_location_2 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(10)
        self.label_location_2.setFont(font)
        self.label_location_2.setObjectName("label_location_2")
        self.horizontalLayout_3.addWidget(self.label_location_2)
        self.lcd_1 = QtWidgets.QLCDNumber(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.lcd_1.setFont(font)
        self.lcd_1.setStyleSheet("QLCDNumber{\n"
" color:black\n"
"}")
        self.lcd_1.setDigitCount(5)
        self.lcd_1.setSegmentStyle(QtWidgets.QLCDNumber.Flat)
        self.lcd_1.setObjectName("lcd_1")
        self.horizontalLayout_3.addWidget(self.lcd_1)
        self.verticalLayout.addLayout(self.horizontalLayout_3)
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.label_location_3 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(10)
        self.label_location_3.setFont(font)
        self.label_location_3.setObjectName("label_location_3")
        self.horizontalLayout_4.addWidget(self.label_location_3)
        self.lcd_2 = QtWidgets.QLCDNumber(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.lcd_2.setFont(font)
        self.lcd_2.setStyleSheet("QLCDNumber{\n"
" color:black\n"
"}")
        self.lcd_2.setSegmentStyle(QtWidgets.QLCDNumber.Flat)
        self.lcd_2.setObjectName("lcd_2")
        self.horizontalLayout_4.addWidget(self.lcd_2)
        self.verticalLayout.addLayout(self.horizontalLayout_4)
        self.horizontalLayout_5.addLayout(self.verticalLayout)
        self.bt_check = QtWidgets.QPushButton(self.centralwidget)
        self.bt_check.setMinimumSize(QtCore.QSize(0, 35))
        font = QtGui.QFont()
        font.setFamily("新宋体")
        font.setPointSize(10)
        self.bt_check.setFont(font)
        self.bt_check.setObjectName("bt_check")
        self.horizontalLayout_5.addWidget(self.bt_check)
        self.verticalLayout_2.addLayout(self.horizontalLayout_5)
        self.horizontalLayout_6 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_6.setObjectName("horizontalLayout_6")
        self.lineEdit_leave = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_leave.setMinimumSize(QtCore.QSize(120, 30))
        font = QtGui.QFont()
        font.setFamily("新宋体")
        font.setPointSize(10)
        self.lineEdit_leave.setFont(font)
        self.lineEdit_leave.setText("")
        self.lineEdit_leave.setObjectName("lineEdit_leave")
        self.horizontalLayout_6.addWidget(self.lineEdit_leave)
        self.bt_leave = QtWidgets.QPushButton(self.centralwidget)
        self.bt_leave.setMinimumSize(QtCore.QSize(100, 35))
        font = QtGui.QFont()
        font.setFamily("新宋体")
        font.setPointSize(10)
        self.bt_leave.setFont(font)
        self.bt_leave.setObjectName("bt_leave")
        self.horizontalLayout_6.addWidget(self.bt_leave)
        self.verticalLayout_2.addLayout(self.horizontalLayout_6)
        self.horizontalLayout_7 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_7.setObjectName("horizontalLayout_7")
        self.lineEdit_supplement = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_supplement.setMinimumSize(QtCore.QSize(120, 30))
        font = QtGui.QFont()
        font.setFamily("新宋体")
        font.setPointSize(10)
        self.lineEdit_supplement.setFont(font)
        self.lineEdit_supplement.setObjectName("lineEdit_supplement")
        self.horizontalLayout_7.addWidget(self.lineEdit_supplement)
        self.bt_supplement = QtWidgets.QPushButton(self.centralwidget)
        self.bt_supplement.setMinimumSize(QtCore.QSize(100, 35))
        font = QtGui.QFont()
        font.setFamily("新宋体")
        font.setPointSize(10)
        self.bt_supplement.setFont(font)
        self.bt_supplement.setObjectName("bt_supplement")
        self.horizontalLayout_7.addWidget(self.bt_supplement)
        self.verticalLayout_2.addLayout(self.horizontalLayout_7)
        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox.setMinimumSize(QtCore.QSize(0, 170))
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(12)
        self.groupBox.setFont(font)
        self.groupBox.setObjectName("groupBox")
        self.horizontalLayout_10 = QtWidgets.QHBoxLayout(self.groupBox)
        self.horizontalLayout_10.setObjectName("horizontalLayout_10")
        self.tableWidget_absent_students = QtWidgets.QTableWidget(self.groupBox)
        font = QtGui.QFont()
        font.setFamily("新宋体")
        font.setPointSize(12)
        self.tableWidget_absent_students.setFont(font)
        self.tableWidget_absent_students.setObjectName("tableWidget_absent_students")
        self.tableWidget_absent_students.setColumnCount(2)
        self.tableWidget_absent_students.setRowCount(0)
        item = QtWidgets.QTableWidgetItem()
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(14)
        item.setFont(font)
        self.tableWidget_absent_students.setHorizontalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(14)
        item.setFont(font)
        self.tableWidget_absent_students.setHorizontalHeaderItem(1, item)
        self.horizontalLayout_10.addWidget(self.tableWidget_absent_students)
        self.verticalLayout_2.addWidget(self.groupBox)
        self.bt_view = QtWidgets.QPushButton(self.centralwidget)
        self.bt_view.setMinimumSize(QtCore.QSize(0, 50))
        font = QtGui.QFont()
        font.setFamily("新宋体")
        font.setPointSize(10)
        self.bt_view.setFont(font)
        self.bt_view.setObjectName("bt_view")
        self.verticalLayout_2.addWidget(self.bt_view)
        self.horizontalLayout_12.addLayout(self.verticalLayout_2)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1136, 26))
        self.menubar.setObjectName("menubar")
        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        self.menuEdit = QtWidgets.QMenu(self.menubar)
        self.menuEdit.setObjectName("menuEdit")
        self.menuView = QtWidgets.QMenu(self.menubar)
        self.menuView.setObjectName("menuView")
        self.menuHelp = QtWidgets.QMenu(self.menubar)
        self.menuHelp.setObjectName("menuHelp")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actionAbout = QtWidgets.QAction(MainWindow)
        self.actionAbout.setObjectName("actionAbout")
        self.actionOpen = QtWidgets.QAction(MainWindow)
        self.actionOpen.setObjectName("actionOpen")
        self.actionSetting = QtWidgets.QAction(MainWindow)
        self.actionSetting.setObjectName("actionSetting")
        self.actionExport = QtWidgets.QAction(MainWindow)
        self.actionExport.setObjectName("actionExport")
        self.menuFile.addAction(self.actionOpen)
        self.menuFile.addAction(self.actionSetting)
        self.menuFile.addAction(self.actionExport)
        self.menuHelp.addAction(self.actionAbout)
        self.menubar.addAction(self.menuFile.menuAction())
        self.menubar.addAction(self.menuEdit.menuAction())
        self.menubar.addAction(self.menuView.menuAction())
        self.menubar.addAction(self.menuHelp.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "考勤"))
        self.textBrowser_log.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'新宋体\'; font-size:15pt; font-weight:400; font-style:normal;\">\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px; font-size:13pt;\"><br /></p></body></html>"))
        self.label_class.setText(_translate("MainWindow", "考勤班级："))
        self.label_location.setText(_translate("MainWindow", "考勤地点："))
        self.label.setText(_translate("MainWindow", "考勤课程："))
        self.bt_open_camera.setText(_translate("MainWindow", "打开摄像头，开始考勤"))
        self.rewrite_infos_btn.setText(_translate("MainWindow", "关闭摄像头，重新填写"))
        self.label_location_2.setText(_translate("MainWindow", "应到："))
        self.label_location_3.setText(_translate("MainWindow", "实到："))
        self.bt_check.setText(_translate("MainWindow", "查询"))
        self.bt_leave.setText(_translate("MainWindow", "请假登记"))
        self.bt_supplement.setText(_translate("MainWindow", "漏签补签"))
        self.groupBox.setTitle(_translate("MainWindow", "未签到"))
        item = self.tableWidget_absent_students.horizontalHeaderItem(0)
        item.setText(_translate("MainWindow", "姓名"))
        item = self.tableWidget_absent_students.horizontalHeaderItem(1)
        item.setText(_translate("MainWindow", "班级"))
        self.bt_view.setText(_translate("MainWindow", "查看结果"))
        self.menuFile.setTitle(_translate("MainWindow", "File"))
        self.menuEdit.setTitle(_translate("MainWindow", "Edit"))
        self.menuView.setTitle(_translate("MainWindow", "View"))
        self.menuHelp.setTitle(_translate("MainWindow", "Help"))
        self.actionAbout.setText(_translate("MainWindow", "About"))
        self.actionAbout.setWhatsThis(_translate("MainWindow", "Author: datamonday"))
        self.actionOpen.setText(_translate("MainWindow", "Open"))
        self.actionSetting.setText(_translate("MainWindow", "Setting"))
        self.actionExport.setText(_translate("MainWindow", "Export"))
