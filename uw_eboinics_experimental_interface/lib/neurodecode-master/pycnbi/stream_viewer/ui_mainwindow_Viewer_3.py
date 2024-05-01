# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'mainwindow.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg
class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1105, 669)
        self.centralWidget = QtWidgets.QWidget(MainWindow)
        self.centralWidget.setObjectName("centralWidget")
        self.comboBox_scale = QtWidgets.QComboBox(self.centralWidget)
        self.comboBox_scale.setGeometry(QtCore.QRect(10, 20, 85, 27))
        self.comboBox_scale.setObjectName("comboBox_scale")
        self.comboBox_scale.addItem("")
        self.comboBox_scale.addItem("")
        self.comboBox_scale.addItem("")
        self.comboBox_scale.addItem("")
        self.comboBox_scale.addItem("")
        self.comboBox_scale.addItem("")
        self.comboBox_scale.addItem("")
        self.comboBox_scale.addItem("")
        self.comboBox_scale.addItem("")
        self.comboBox_scale.addItem("")
        self.checkBox_car = QtWidgets.QCheckBox(self.centralWidget)
        self.checkBox_car.setGeometry(QtCore.QRect(120, 0, 97, 22))
        self.checkBox_car.setObjectName("checkBox_car")
        self.label_2 = QtWidgets.QLabel(self.centralWidget)
        self.label_2.setGeometry(QtCore.QRect(10, 60, 67, 17))
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(self.centralWidget)
        self.label_3.setGeometry(QtCore.QRect(10, 0, 67, 17))
        self.label_3.setObjectName("label_3")
        self.checkBox_bandpass = QtWidgets.QCheckBox(self.centralWidget)
        self.checkBox_bandpass.setGeometry(QtCore.QRect(120, 20, 141, 22))
        self.checkBox_bandpass.setObjectName("checkBox_bandpass")
        self.pushButton_bp = QtWidgets.QPushButton(self.centralWidget)
        self.pushButton_bp.setGeometry(QtCore.QRect(410, 20, 99, 27))
        self.pushButton_bp.setObjectName("pushButton_bp")
        self.spinBox_time = QtWidgets.QSpinBox(self.centralWidget)
        self.spinBox_time.setGeometry(QtCore.QRect(10, 80, 85, 27))
        self.spinBox_time.setMinimum(1)
        self.spinBox_time.setProperty("value", 5)
        self.spinBox_time.setObjectName("spinBox_time")
        self.doubleSpinBox_hp = QtWidgets.QDoubleSpinBox(self.centralWidget)
        self.doubleSpinBox_hp.setGeometry(QtCore.QRect(260, 20, 69, 27))
        self.doubleSpinBox_hp.setObjectName("doubleSpinBox_hp")
        self.doubleSpinBox_lp = QtWidgets.QDoubleSpinBox(self.centralWidget)
        self.doubleSpinBox_lp.setGeometry(QtCore.QRect(340, 20, 69, 27))
        self.doubleSpinBox_lp.setObjectName("doubleSpinBox_lp")
        self.checkBox_showTID = QtWidgets.QCheckBox(self.centralWidget)
        self.checkBox_showTID.setGeometry(QtCore.QRect(120, 60, 161, 22))
        self.checkBox_showTID.setObjectName("checkBox_showTID")
        self.checkBox_showLPT = QtWidgets.QCheckBox(self.centralWidget)
        self.checkBox_showLPT.setGeometry(QtCore.QRect(120, 80, 151, 22))
        self.checkBox_showLPT.setObjectName("checkBox_showLPT")
        self.checkBox_showKey = QtWidgets.QCheckBox(self.centralWidget)
        self.checkBox_showKey.setGeometry(QtCore.QRect(120, 100, 151, 22))
        self.checkBox_showKey.setObjectName("checkBox_showKey")
        self.line = QtWidgets.QFrame(self.centralWidget)
        self.line.setGeometry(QtCore.QRect(100, 0, 21, 121))
        self.line.setFrameShape(QtWidgets.QFrame.VLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.line_2 = QtWidgets.QFrame(self.centralWidget)
        self.line_2.setGeometry(QtCore.QRect(110, 45, 431, 21))
        self.line_2.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_2.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_2.setObjectName("line_2")
        self.line_3 = QtWidgets.QFrame(self.centralWidget)
        self.line_3.setGeometry(QtCore.QRect(290, 60, 21, 61))
        self.line_3.setFrameShape(QtWidgets.QFrame.VLine)
        self.line_3.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_3.setObjectName("line_3")
        self.pushButton_stoprec = QtWidgets.QPushButton(self.centralWidget)
        self.pushButton_stoprec.setGeometry(QtCore.QRect(450, 90, 61, 27))
        self.pushButton_stoprec.setObjectName("pushButton_stoprec")
        self.pushButton_rec = QtWidgets.QPushButton(self.centralWidget)
        self.pushButton_rec.setGeometry(QtCore.QRect(460, 60, 51, 27))
        self.pushButton_rec.setObjectName("pushButton_rec")
        self.lineEdit_recFilename = QtWidgets.QLineEdit(self.centralWidget)
        self.lineEdit_recFilename.setGeometry(QtCore.QRect(310, 60, 141, 27))
        self.lineEdit_recFilename.setObjectName("lineEdit_recFilename")
        self.table_channels = QtWidgets.QTableWidget(self.centralWidget)
        self.table_channels.setGeometry(QtCore.QRect(60, 140, 391, 501))
        self.table_channels.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table_channels.setTextElideMode(QtCore.Qt.ElideMiddle)
        self.table_channels.setShowGrid(False)
        self.table_channels.setObjectName("table_channels")
        self.table_channels.setColumnCount(4)
        self.table_channels.setRowCount(16)
        item = QtWidgets.QTableWidgetItem()
        self.table_channels.setVerticalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        self.table_channels.setVerticalHeaderItem(1, item)
        item = QtWidgets.QTableWidgetItem()
        self.table_channels.setVerticalHeaderItem(2, item)
        item = QtWidgets.QTableWidgetItem()
        self.table_channels.setVerticalHeaderItem(3, item)
        item = QtWidgets.QTableWidgetItem()
        self.table_channels.setVerticalHeaderItem(4, item)
        item = QtWidgets.QTableWidgetItem()
        self.table_channels.setVerticalHeaderItem(5, item)
        item = QtWidgets.QTableWidgetItem()
        self.table_channels.setVerticalHeaderItem(6, item)
        item = QtWidgets.QTableWidgetItem()
        self.table_channels.setVerticalHeaderItem(7, item)
        item = QtWidgets.QTableWidgetItem()
        self.table_channels.setVerticalHeaderItem(8, item)
        item = QtWidgets.QTableWidgetItem()
        self.table_channels.setVerticalHeaderItem(9, item)
        item = QtWidgets.QTableWidgetItem()
        self.table_channels.setVerticalHeaderItem(10, item)
        item = QtWidgets.QTableWidgetItem()
        self.table_channels.setVerticalHeaderItem(11, item)
        item = QtWidgets.QTableWidgetItem()
        self.table_channels.setVerticalHeaderItem(12, item)
        item = QtWidgets.QTableWidgetItem()
        self.table_channels.setVerticalHeaderItem(13, item)
        item = QtWidgets.QTableWidgetItem()
        self.table_channels.setVerticalHeaderItem(14, item)
        item = QtWidgets.QTableWidgetItem()
        self.table_channels.setVerticalHeaderItem(15, item)
        item = QtWidgets.QTableWidgetItem()
        self.table_channels.setHorizontalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        self.table_channels.setHorizontalHeaderItem(1, item)
        item = QtWidgets.QTableWidgetItem()
        self.table_channels.setHorizontalHeaderItem(2, item)
        item = QtWidgets.QTableWidgetItem()
        self.table_channels.setHorizontalHeaderItem(3, item)
        item = QtWidgets.QTableWidgetItem()
        self.table_channels.setItem(0, 0, item)
        item = QtWidgets.QTableWidgetItem()
        self.table_channels.setItem(0, 1, item)
        item = QtWidgets.QTableWidgetItem()
        self.table_channels.setItem(0, 2, item)
        item = QtWidgets.QTableWidgetItem()
        self.table_channels.setItem(0, 3, item)
        item = QtWidgets.QTableWidgetItem()
        self.table_channels.setItem(1, 0, item)
        item = QtWidgets.QTableWidgetItem()
        self.table_channels.setItem(1, 1, item)
        item = QtWidgets.QTableWidgetItem()
        self.table_channels.setItem(1, 2, item)
        item = QtWidgets.QTableWidgetItem()
        self.table_channels.setItem(1, 3, item)
        item = QtWidgets.QTableWidgetItem()
        self.table_channels.setItem(2, 0, item)
        item = QtWidgets.QTableWidgetItem()
        self.table_channels.setItem(2, 1, item)
        item = QtWidgets.QTableWidgetItem()
        self.table_channels.setItem(2, 2, item)
        item = QtWidgets.QTableWidgetItem()
        self.table_channels.setItem(2, 3, item)
        item = QtWidgets.QTableWidgetItem()
        self.table_channels.setItem(3, 0, item)
        item = QtWidgets.QTableWidgetItem()
        self.table_channels.setItem(3, 1, item)
        item = QtWidgets.QTableWidgetItem()
        self.table_channels.setItem(3, 2, item)
        item = QtWidgets.QTableWidgetItem()
        self.table_channels.setItem(3, 3, item)
        item = QtWidgets.QTableWidgetItem()
        self.table_channels.setItem(4, 0, item)
        item = QtWidgets.QTableWidgetItem()
        self.table_channels.setItem(4, 1, item)
        item = QtWidgets.QTableWidgetItem()
        self.table_channels.setItem(4, 2, item)
        item = QtWidgets.QTableWidgetItem()
        self.table_channels.setItem(4, 3, item)
        item = QtWidgets.QTableWidgetItem()
        self.table_channels.setItem(5, 0, item)
        item = QtWidgets.QTableWidgetItem()
        self.table_channels.setItem(5, 1, item)
        item = QtWidgets.QTableWidgetItem()
        self.table_channels.setItem(5, 2, item)
        item = QtWidgets.QTableWidgetItem()
        self.table_channels.setItem(5, 3, item)
        item = QtWidgets.QTableWidgetItem()
        self.table_channels.setItem(6, 0, item)
        item = QtWidgets.QTableWidgetItem()
        self.table_channels.setItem(6, 1, item)
        item = QtWidgets.QTableWidgetItem()
        self.table_channels.setItem(6, 2, item)
        item = QtWidgets.QTableWidgetItem()
        self.table_channels.setItem(6, 3, item)
        item = QtWidgets.QTableWidgetItem()
        self.table_channels.setItem(7, 0, item)
        item = QtWidgets.QTableWidgetItem()
        self.table_channels.setItem(7, 1, item)
        item = QtWidgets.QTableWidgetItem()
        self.table_channels.setItem(7, 2, item)
        item = QtWidgets.QTableWidgetItem()
        self.table_channels.setItem(7, 3, item)
        item = QtWidgets.QTableWidgetItem()
        self.table_channels.setItem(8, 0, item)
        item = QtWidgets.QTableWidgetItem()
        self.table_channels.setItem(8, 1, item)
        item = QtWidgets.QTableWidgetItem()
        self.table_channels.setItem(8, 2, item)
        item = QtWidgets.QTableWidgetItem()
        self.table_channels.setItem(8, 3, item)
        item = QtWidgets.QTableWidgetItem()
        self.table_channels.setItem(9, 0, item)
        item = QtWidgets.QTableWidgetItem()
        self.table_channels.setItem(9, 1, item)
        item = QtWidgets.QTableWidgetItem()
        self.table_channels.setItem(9, 2, item)
        item = QtWidgets.QTableWidgetItem()
        self.table_channels.setItem(9, 3, item)
        item = QtWidgets.QTableWidgetItem()
        self.table_channels.setItem(10, 0, item)
        item = QtWidgets.QTableWidgetItem()
        self.table_channels.setItem(10, 1, item)
        item = QtWidgets.QTableWidgetItem()
        self.table_channels.setItem(10, 2, item)
        item = QtWidgets.QTableWidgetItem()
        self.table_channels.setItem(10, 3, item)
        item = QtWidgets.QTableWidgetItem()
        self.table_channels.setItem(11, 0, item)
        item = QtWidgets.QTableWidgetItem()
        self.table_channels.setItem(11, 1, item)
        item = QtWidgets.QTableWidgetItem()
        self.table_channels.setItem(11, 2, item)
        item = QtWidgets.QTableWidgetItem()
        self.table_channels.setItem(11, 3, item)
        item = QtWidgets.QTableWidgetItem()
        self.table_channels.setItem(12, 0, item)
        item = QtWidgets.QTableWidgetItem()
        self.table_channels.setItem(12, 1, item)
        item = QtWidgets.QTableWidgetItem()
        self.table_channels.setItem(12, 2, item)
        item = QtWidgets.QTableWidgetItem()
        self.table_channels.setItem(12, 3, item)
        item = QtWidgets.QTableWidgetItem()
        self.table_channels.setItem(13, 0, item)
        item = QtWidgets.QTableWidgetItem()
        self.table_channels.setItem(13, 1, item)
        item = QtWidgets.QTableWidgetItem()
        self.table_channels.setItem(13, 2, item)
        item = QtWidgets.QTableWidgetItem()
        self.table_channels.setItem(13, 3, item)
        item = QtWidgets.QTableWidgetItem()
        self.table_channels.setItem(14, 0, item)
        item = QtWidgets.QTableWidgetItem()
        self.table_channels.setItem(14, 1, item)
        item = QtWidgets.QTableWidgetItem()
        self.table_channels.setItem(14, 2, item)
        item = QtWidgets.QTableWidgetItem()
        self.table_channels.setItem(14, 3, item)
        item = QtWidgets.QTableWidgetItem()
        self.table_channels.setItem(15, 0, item)
        item = QtWidgets.QTableWidgetItem()
        self.table_channels.setItem(15, 1, item)
        item = QtWidgets.QTableWidgetItem()
        self.table_channels.setItem(15, 2, item)
        item = QtWidgets.QTableWidgetItem()
        self.table_channels.setItem(15, 3, item)
        self.table_channels.horizontalHeader().setVisible(False)
        self.table_channels.horizontalHeader().setHighlightSections(False)
        self.table_channels.verticalHeader().setVisible(False)
        self.table_channels.verticalHeader().setHighlightSections(False)
        self.line_4 = QtWidgets.QFrame(self.centralWidget)
        self.line_4.setGeometry(QtCore.QRect(0, 120, 531, 21))
        self.line_4.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_4.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_4.setObjectName("line_4")
        self.lcdNumber = QtWidgets.QLCDNumber(self.centralWidget)
        self.lcdNumber.setGeometry(QtCore.QRect(540, 60, 64, 23))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.lcdNumber.setFont(font)
        self.lcdNumber.setObjectName("lcdNumber")
        self.pushButton_reset = QtWidgets.QPushButton(self.centralWidget)
        self.pushButton_reset.setGeometry(QtCore.QRect(540, 90, 75, 23))
        self.pushButton_reset.setObjectName("pushButton_reset")
        self.pushButton_start_SV = QtWidgets.QPushButton(self.centralWidget)
        self.pushButton_start_SV.setGeometry(QtCore.QRect(630, 60, 101, 31))
        self.pushButton_start_SV.setObjectName("pushButton_start_SV")
        self.pushButton_stop_SV = QtWidgets.QPushButton(self.centralWidget)
        self.pushButton_stop_SV.setGeometry(QtCore.QRect(630, 100, 101, 31))
        self.pushButton_stop_SV.setObjectName("pushButton_stop_SV")
        self.pushButton_start_train = QtWidgets.QPushButton(self.centralWidget)
        self.pushButton_start_train.setGeometry(QtCore.QRect(640, 150, 75, 23))
        self.pushButton_start_train.setObjectName("pushButton_start_train")
        self.pushButton_stop_train = QtWidgets.QPushButton(self.centralWidget)
        self.pushButton_stop_train.setGeometry(QtCore.QRect(640, 190, 75, 23))
        self.pushButton_stop_train.setObjectName("pushButton_stop_train")
        self.pushButton_start_test = QtWidgets.QPushButton(self.centralWidget)
        self.pushButton_start_test.setGeometry(QtCore.QRect(640, 230, 75, 23))
        self.pushButton_start_test.setObjectName("pushButton_start_test")
        self.pushButton_stop_test = QtWidgets.QPushButton(self.centralWidget)
        self.pushButton_stop_test.setGeometry(QtCore.QRect(640, 270, 75, 23))
        self.pushButton_stop_test.setObjectName("pushButton_stop_test")
        #self.graphicsView = QtWidgets.QGraphicsView(self.centralWidget)
        self.graphicsView = pg.PlotWidget(self.centralWidget)
        self.graphicsView.setGeometry(QtCore.QRect(780, 60, 311, 201))
        self.graphicsView.setObjectName("graphicsView")
        MainWindow.setCentralWidget(self.centralWidget)
        self.menuBar = QtWidgets.QMenuBar(MainWindow)
        self.menuBar.setGeometry(QtCore.QRect(0, 0, 1105, 21))
        self.menuBar.setObjectName("menuBar")
        MainWindow.setMenuBar(self.menuBar)
        self.mainToolBar = QtWidgets.QToolBar(MainWindow)
        self.mainToolBar.setObjectName("mainToolBar")
        MainWindow.addToolBar(QtCore.Qt.TopToolBarArea, self.mainToolBar)
        self.statusBar = QtWidgets.QStatusBar(MainWindow)
        self.statusBar.setObjectName("statusBar")
        MainWindow.setStatusBar(self.statusBar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.comboBox_scale.setItemText(0, _translate("MainWindow", "1uV"))
        self.comboBox_scale.setItemText(1, _translate("MainWindow", "10uV"))
        self.comboBox_scale.setItemText(2, _translate("MainWindow", "25uV"))
        self.comboBox_scale.setItemText(3, _translate("MainWindow", "50uV"))
        self.comboBox_scale.setItemText(4, _translate("MainWindow", "100uV"))
        self.comboBox_scale.setItemText(5, _translate("MainWindow", "250uV"))
        self.comboBox_scale.setItemText(6, _translate("MainWindow", "500uV"))
        self.comboBox_scale.setItemText(7, _translate("MainWindow", "1mV"))
        self.comboBox_scale.setItemText(8, _translate("MainWindow", "2.5mV"))
        self.comboBox_scale.setItemText(9, _translate("MainWindow", "100mV"))
        self.checkBox_car.setText(_translate("MainWindow", "CAR Filter"))
        self.label_2.setText(_translate("MainWindow", "Time (s)"))
        self.label_3.setText(_translate("MainWindow", "Scale"))
        self.checkBox_bandpass.setText(_translate("MainWindow", "Bandpass Filter"))
        self.pushButton_bp.setText(_translate("MainWindow", "Apply BP"))
        self.checkBox_showTID.setText(_translate("MainWindow", "Show TID events"))
        self.checkBox_showLPT.setText(_translate("MainWindow", "Show LPT events"))
        self.checkBox_showKey.setText(_translate("MainWindow", "Show Key events"))
        self.pushButton_stoprec.setText(_translate("MainWindow", "Stop REC"))
        self.pushButton_rec.setText(_translate("MainWindow", "REC"))
        item = self.table_channels.verticalHeaderItem(0)
        item.setText(_translate("MainWindow", "1"))
        item = self.table_channels.verticalHeaderItem(1)
        item.setText(_translate("MainWindow", "2"))
        item = self.table_channels.verticalHeaderItem(2)
        item.setText(_translate("MainWindow", "3"))
        item = self.table_channels.verticalHeaderItem(3)
        item.setText(_translate("MainWindow", "4"))
        item = self.table_channels.verticalHeaderItem(4)
        item.setText(_translate("MainWindow", "5"))
        item = self.table_channels.verticalHeaderItem(5)
        item.setText(_translate("MainWindow", "6"))
        item = self.table_channels.verticalHeaderItem(6)
        item.setText(_translate("MainWindow", "7"))
        item = self.table_channels.verticalHeaderItem(7)
        item.setText(_translate("MainWindow", "8"))
        item = self.table_channels.verticalHeaderItem(8)
        item.setText(_translate("MainWindow", "9"))
        item = self.table_channels.verticalHeaderItem(9)
        item.setText(_translate("MainWindow", "10"))
        item = self.table_channels.verticalHeaderItem(10)
        item.setText(_translate("MainWindow", "11"))
        item = self.table_channels.verticalHeaderItem(11)
        item.setText(_translate("MainWindow", "12"))
        item = self.table_channels.verticalHeaderItem(12)
        item.setText(_translate("MainWindow", "13"))
        item = self.table_channels.verticalHeaderItem(13)
        item.setText(_translate("MainWindow", "14"))
        item = self.table_channels.verticalHeaderItem(14)
        item.setText(_translate("MainWindow", "15"))
        item = self.table_channels.verticalHeaderItem(15)
        item.setText(_translate("MainWindow", "16"))
        item = self.table_channels.horizontalHeaderItem(0)
        item.setText(_translate("MainWindow", "1"))
        item = self.table_channels.horizontalHeaderItem(1)
        item.setText(_translate("MainWindow", "2"))
        item = self.table_channels.horizontalHeaderItem(2)
        item.setText(_translate("MainWindow", "3"))
        item = self.table_channels.horizontalHeaderItem(3)
        item.setText(_translate("MainWindow", "4"))
        __sortingEnabled = self.table_channels.isSortingEnabled()
        self.table_channels.setSortingEnabled(False)
        item = self.table_channels.item(0, 0)
        item.setText(_translate("MainWindow", "1"))
        item = self.table_channels.item(0, 1)
        item.setText(_translate("MainWindow", "17"))
        item = self.table_channels.item(0, 2)
        item.setText(_translate("MainWindow", "33"))
        item = self.table_channels.item(0, 3)
        item.setText(_translate("MainWindow", "49"))
        item = self.table_channels.item(1, 0)
        item.setText(_translate("MainWindow", "2"))
        item = self.table_channels.item(1, 1)
        item.setText(_translate("MainWindow", "18"))
        item = self.table_channels.item(1, 2)
        item.setText(_translate("MainWindow", "34"))
        item = self.table_channels.item(1, 3)
        item.setText(_translate("MainWindow", "50"))
        item = self.table_channels.item(2, 0)
        item.setText(_translate("MainWindow", "3"))
        item = self.table_channels.item(2, 1)
        item.setText(_translate("MainWindow", "19"))
        item = self.table_channels.item(2, 2)
        item.setText(_translate("MainWindow", "35"))
        item = self.table_channels.item(2, 3)
        item.setText(_translate("MainWindow", "51"))
        item = self.table_channels.item(3, 0)
        item.setText(_translate("MainWindow", "4"))
        item = self.table_channels.item(3, 1)
        item.setText(_translate("MainWindow", "20"))
        item = self.table_channels.item(3, 2)
        item.setText(_translate("MainWindow", "36"))
        item = self.table_channels.item(3, 3)
        item.setText(_translate("MainWindow", "52"))
        item = self.table_channels.item(4, 0)
        item.setText(_translate("MainWindow", "5"))
        item = self.table_channels.item(4, 1)
        item.setText(_translate("MainWindow", "21"))
        item = self.table_channels.item(4, 2)
        item.setText(_translate("MainWindow", "37"))
        item = self.table_channels.item(4, 3)
        item.setText(_translate("MainWindow", "53"))
        item = self.table_channels.item(5, 0)
        item.setText(_translate("MainWindow", "6"))
        item = self.table_channels.item(5, 1)
        item.setText(_translate("MainWindow", "22"))
        item = self.table_channels.item(5, 2)
        item.setText(_translate("MainWindow", "38"))
        item = self.table_channels.item(5, 3)
        item.setText(_translate("MainWindow", "54"))
        item = self.table_channels.item(6, 0)
        item.setText(_translate("MainWindow", "7"))
        item = self.table_channels.item(6, 1)
        item.setText(_translate("MainWindow", "23"))
        item = self.table_channels.item(6, 2)
        item.setText(_translate("MainWindow", "39"))
        item = self.table_channels.item(6, 3)
        item.setText(_translate("MainWindow", "55"))
        item = self.table_channels.item(7, 0)
        item.setText(_translate("MainWindow", "8"))
        item = self.table_channels.item(7, 1)
        item.setText(_translate("MainWindow", "24"))
        item = self.table_channels.item(7, 2)
        item.setText(_translate("MainWindow", "40"))
        item = self.table_channels.item(7, 3)
        item.setText(_translate("MainWindow", "56"))
        item = self.table_channels.item(8, 0)
        item.setText(_translate("MainWindow", "9"))
        item = self.table_channels.item(8, 1)
        item.setText(_translate("MainWindow", "25"))
        item = self.table_channels.item(8, 2)
        item.setText(_translate("MainWindow", "41"))
        item = self.table_channels.item(8, 3)
        item.setText(_translate("MainWindow", "57"))
        item = self.table_channels.item(9, 0)
        item.setText(_translate("MainWindow", "10"))
        item = self.table_channels.item(9, 1)
        item.setText(_translate("MainWindow", "26"))
        item = self.table_channels.item(9, 2)
        item.setText(_translate("MainWindow", "42"))
        item = self.table_channels.item(9, 3)
        item.setText(_translate("MainWindow", "58"))
        item = self.table_channels.item(10, 0)
        item.setText(_translate("MainWindow", "11"))
        item = self.table_channels.item(10, 1)
        item.setText(_translate("MainWindow", "27"))
        item = self.table_channels.item(10, 2)
        item.setText(_translate("MainWindow", "43"))
        item = self.table_channels.item(10, 3)
        item.setText(_translate("MainWindow", "59"))
        item = self.table_channels.item(11, 0)
        item.setText(_translate("MainWindow", "12"))
        item = self.table_channels.item(11, 1)
        item.setText(_translate("MainWindow", "28"))
        item = self.table_channels.item(11, 2)
        item.setText(_translate("MainWindow", "44"))
        item = self.table_channels.item(11, 3)
        item.setText(_translate("MainWindow", "60"))
        item = self.table_channels.item(12, 0)
        item.setText(_translate("MainWindow", "13"))
        item = self.table_channels.item(12, 1)
        item.setText(_translate("MainWindow", "29"))
        item = self.table_channels.item(12, 2)
        item.setText(_translate("MainWindow", "45"))
        item = self.table_channels.item(12, 3)
        item.setText(_translate("MainWindow", "61"))
        item = self.table_channels.item(13, 0)
        item.setText(_translate("MainWindow", "14"))
        item = self.table_channels.item(13, 1)
        item.setText(_translate("MainWindow", "30"))
        item = self.table_channels.item(13, 2)
        item.setText(_translate("MainWindow", "46"))
        item = self.table_channels.item(13, 3)
        item.setText(_translate("MainWindow", "62"))
        item = self.table_channels.item(14, 0)
        item.setText(_translate("MainWindow", "15"))
        item = self.table_channels.item(14, 1)
        item.setText(_translate("MainWindow", "31"))
        item = self.table_channels.item(14, 2)
        item.setText(_translate("MainWindow", "47"))
        item = self.table_channels.item(14, 3)
        item.setText(_translate("MainWindow", "63"))
        item = self.table_channels.item(15, 0)
        item.setText(_translate("MainWindow", "16"))
        item = self.table_channels.item(15, 1)
        item.setText(_translate("MainWindow", "32"))
        item = self.table_channels.item(15, 2)
        item.setText(_translate("MainWindow", "48"))
        item = self.table_channels.item(15, 3)
        item.setText(_translate("MainWindow", "64"))
        self.table_channels.setSortingEnabled(__sortingEnabled)
        self.pushButton_reset.setText(_translate("MainWindow", "reset"))
        self.pushButton_start_SV.setText(_translate("MainWindow", "Start experiment"))
        self.pushButton_stop_SV.setText(_translate("MainWindow", "Stop experiment"))
        self.pushButton_start_train.setText(_translate("MainWindow", "start train"))
        self.pushButton_stop_train.setText(_translate("MainWindow", "stop train"))
        self.pushButton_start_test.setText(_translate("MainWindow", "start test"))
        self.pushButton_stop_test.setText(_translate("MainWindow", "stop test"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

