# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'MatrixWinUi.ui'
#
# Created by: PyQt5 UI code generator 5.8.1
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MatrixWin(object):
    def setupUi(self, MatrixWin):
        MatrixWin.setObjectName("MatrixWin")
        MatrixWin.resize(742, 461)
        self.groupBox = QtWidgets.QGroupBox(MatrixWin)
        self.groupBox.setGeometry(QtCore.QRect(10, 210, 451, 191))
        self.groupBox.setObjectName("groupBox")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.groupBox)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.speedButton1 = QtWidgets.QRadioButton(self.groupBox)
        self.speedButton1.setObjectName("speedButton1")
        self.speedButtonGroup = QtWidgets.QButtonGroup(MatrixWin)
        self.speedButtonGroup.setObjectName("speedButtonGroup")
        self.speedButtonGroup.addButton(self.speedButton1)
        self.gridLayout_2.addWidget(self.speedButton1, 0, 0, 1, 1)
        self.speedButton3 = QtWidgets.QRadioButton(self.groupBox)
        self.speedButton3.setObjectName("speedButton3")
        self.speedButtonGroup.addButton(self.speedButton3)
        self.gridLayout_2.addWidget(self.speedButton3, 0, 2, 1, 1)
        self.speedButton4 = QtWidgets.QRadioButton(self.groupBox)
        self.speedButton4.setObjectName("speedButton4")
        self.speedButtonGroup.addButton(self.speedButton4)
        self.gridLayout_2.addWidget(self.speedButton4, 1, 0, 1, 1)
        self.speedButton5 = QtWidgets.QRadioButton(self.groupBox)
        self.speedButton5.setChecked(True)
        self.speedButton5.setObjectName("speedButton5")
        self.speedButtonGroup.addButton(self.speedButton5)
        self.gridLayout_2.addWidget(self.speedButton5, 1, 1, 1, 1)
        self.speedButton6 = QtWidgets.QRadioButton(self.groupBox)
        self.speedButton6.setObjectName("speedButton6")
        self.speedButtonGroup.addButton(self.speedButton6)
        self.gridLayout_2.addWidget(self.speedButton6, 1, 2, 1, 1)
        self.speedButton9 = QtWidgets.QRadioButton(self.groupBox)
        self.speedButton9.setObjectName("speedButton9")
        self.speedButtonGroup.addButton(self.speedButton9)
        self.gridLayout_2.addWidget(self.speedButton9, 3, 2, 1, 1)
        self.speedButton8 = QtWidgets.QRadioButton(self.groupBox)
        self.speedButton8.setObjectName("speedButton8")
        self.speedButtonGroup.addButton(self.speedButton8)
        self.gridLayout_2.addWidget(self.speedButton8, 3, 1, 1, 1)
        self.speedButton7 = QtWidgets.QRadioButton(self.groupBox)
        self.speedButton7.setObjectName("speedButton7")
        self.speedButtonGroup.addButton(self.speedButton7)
        self.gridLayout_2.addWidget(self.speedButton7, 3, 0, 1, 1)
        self.speedButton2 = QtWidgets.QRadioButton(self.groupBox)
        self.speedButton2.setObjectName("speedButton2")
        self.speedButtonGroup.addButton(self.speedButton2)
        self.gridLayout_2.addWidget(self.speedButton2, 0, 1, 1, 1)
        self.resultGroup = QtWidgets.QGroupBox(MatrixWin)
        self.resultGroup.setGeometry(QtCore.QRect(470, 210, 261, 191))
        self.resultGroup.setObjectName("resultGroup")
        self.resultText = QtWidgets.QTextEdit(self.resultGroup)
        self.resultText.setGeometry(QtCore.QRect(10, 20, 241, 161))
        self.resultText.setObjectName("resultText")
        self.layoutWidget = QtWidgets.QWidget(MatrixWin)
        self.layoutWidget.setGeometry(QtCore.QRect(10, 420, 390, 30))
        self.layoutWidget.setObjectName("layoutWidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.layoutWidget)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.okBtn = QtWidgets.QPushButton(self.layoutWidget)
        self.okBtn.setObjectName("okBtn")
        self.horizontalLayout.addWidget(self.okBtn)
        self.clearBtn = QtWidgets.QPushButton(self.layoutWidget)
        self.clearBtn.setObjectName("clearBtn")
        self.horizontalLayout.addWidget(self.clearBtn)
        self.cancelBtn = QtWidgets.QPushButton(self.layoutWidget)
        self.cancelBtn.setObjectName("cancelBtn")
        self.horizontalLayout.addWidget(self.cancelBtn)
        self.groupBox_2 = QtWidgets.QGroupBox(MatrixWin)
        self.groupBox_2.setGeometry(QtCore.QRect(10, 10, 721, 191))
        self.groupBox_2.setObjectName("groupBox_2")
        self.label = QtWidgets.QLabel(self.groupBox_2)
        self.label.setGeometry(QtCore.QRect(20, 30, 151, 21))
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.groupBox_2)
        self.label_2.setGeometry(QtCore.QRect(20, 60, 151, 21))
        self.label_2.setObjectName("label_2")
        self.label_7 = QtWidgets.QLabel(self.groupBox_2)
        self.label_7.setGeometry(QtCore.QRect(20, 90, 131, 21))
        self.label_7.setObjectName("label_7")
        self.label_4 = QtWidgets.QLabel(self.groupBox_2)
        self.label_4.setGeometry(QtCore.QRect(20, 120, 151, 22))
        self.label_4.setObjectName("label_4")
        self.tequilaScrollBar = QtWidgets.QScrollBar(self.groupBox_2)
        self.tequilaScrollBar.setEnabled(True)
        self.tequilaScrollBar.setGeometry(QtCore.QRect(130, 30, 361, 21))
        self.tequilaScrollBar.setMaximum(11)
        self.tequilaScrollBar.setProperty("value", 8)
        self.tequilaScrollBar.setSliderPosition(8)
        self.tequilaScrollBar.setOrientation(QtCore.Qt.Horizontal)
        self.tequilaScrollBar.setObjectName("tequilaScrollBar")
        self.tripleSecSpinBox = QtWidgets.QSpinBox(self.groupBox_2)
        self.tripleSecSpinBox.setGeometry(QtCore.QRect(130, 60, 250, 21))
        self.tripleSecSpinBox.setMaximum(11)
        self.tripleSecSpinBox.setProperty("value", 4)
        self.tripleSecSpinBox.setObjectName("tripleSecSpinBox")
        self.limeJuiceLineEdit = QtWidgets.QLineEdit(self.groupBox_2)
        self.limeJuiceLineEdit.setGeometry(QtCore.QRect(130, 90, 257, 21))
        self.limeJuiceLineEdit.setObjectName("limeJuiceLineEdit")
        self.iceHorizontalSlider = QtWidgets.QSlider(self.groupBox_2)
        self.iceHorizontalSlider.setGeometry(QtCore.QRect(130, 120, 250, 22))
        self.iceHorizontalSlider.setMinimum(0)
        self.iceHorizontalSlider.setMaximum(20)
        self.iceHorizontalSlider.setProperty("value", 12)
        self.iceHorizontalSlider.setOrientation(QtCore.Qt.Horizontal)
        self.iceHorizontalSlider.setObjectName("iceHorizontalSlider")
        self.label_6 = QtWidgets.QLabel(self.groupBox_2)
        self.label_6.setGeometry(QtCore.QRect(610, 30, 61, 21))
        self.label_6.setObjectName("label_6")
        self.label_3 = QtWidgets.QLabel(self.groupBox_2)
        self.label_3.setGeometry(QtCore.QRect(610, 50, 61, 21))
        self.label_3.setObjectName("label_3")
        self.label_8 = QtWidgets.QLabel(self.groupBox_2)
        self.label_8.setGeometry(QtCore.QRect(610, 80, 61, 21))
        self.label_8.setObjectName("label_8")
        self.label_5 = QtWidgets.QLabel(self.groupBox_2)
        self.label_5.setGeometry(QtCore.QRect(610, 120, 61, 21))
        self.label_5.setObjectName("label_5")
        self.selScrollBarLbl = QtWidgets.QLabel(self.groupBox_2)
        self.selScrollBarLbl.setGeometry(QtCore.QRect(520, 30, 51, 21))
        self.selScrollBarLbl.setText("")
        self.selScrollBarLbl.setObjectName("selScrollBarLbl")
        self.selIceSliderLbl = QtWidgets.QLabel(self.groupBox_2)
        self.selIceSliderLbl.setGeometry(QtCore.QRect(520, 120, 51, 21))
        self.selIceSliderLbl.setText("")
        self.selIceSliderLbl.setObjectName("selIceSliderLbl")

        self.retranslateUi(MatrixWin)
        self.okBtn.clicked.connect(MatrixWin.uiAccept)
        self.cancelBtn.clicked.connect(MatrixWin.uiReject)
        self.clearBtn.clicked.connect(MatrixWin.uiClear)
        self.iceHorizontalSlider.valueChanged['int'].connect(MatrixWin.uiIceSliderValueChanged)
        self.tequilaScrollBar.valueChanged['int'].connect(MatrixWin.uiScrollBarValueChanged)
        QtCore.QMetaObject.connectSlotsByName(MatrixWin)

    def retranslateUi(self, MatrixWin):
        _translate = QtCore.QCoreApplication.translate
        MatrixWin.setWindowTitle(_translate("MatrixWin", "玛格丽特鸡尾酒*调酒器"))
        self.groupBox.setToolTip(_translate("MatrixWin", "Speed of the blender"))
        self.groupBox.setTitle(_translate("MatrixWin", "9种搅拌速度"))
        self.speedButton1.setText(_translate("MatrixWin", "&Mix"))
        self.speedButton3.setText(_translate("MatrixWin", "&Puree"))
        self.speedButton4.setText(_translate("MatrixWin", "&Chop"))
        self.speedButton5.setText(_translate("MatrixWin", "&Karate Chop"))
        self.speedButton6.setText(_translate("MatrixWin", "&Beat"))
        self.speedButton9.setText(_translate("MatrixWin", "&Vaporize"))
        self.speedButton8.setText(_translate("MatrixWin", "&Liquefy"))
        self.speedButton7.setText(_translate("MatrixWin", "&Smash"))
        self.speedButton2.setText(_translate("MatrixWin", "&Whip"))
        self.resultGroup.setTitle(_translate("MatrixWin", "操作结果"))
        self.okBtn.setText(_translate("MatrixWin", "OK"))
        self.clearBtn.setText(_translate("MatrixWin", "Clear"))
        self.cancelBtn.setText(_translate("MatrixWin", "Cancel"))
        self.groupBox_2.setTitle(_translate("MatrixWin", "原料"))
        self.label.setText(_translate("MatrixWin", "龙舌兰酒"))
        self.label_2.setText(_translate("MatrixWin", "三重蒸馏酒"))
        self.label_7.setText(_translate("MatrixWin", "柠檬汁"))
        self.label_4.setText(_translate("MatrixWin", "冰块"))
        self.tequilaScrollBar.setToolTip(_translate("MatrixWin", "Jiggers of tequila"))
        self.tripleSecSpinBox.setToolTip(_translate("MatrixWin", "Jiggers of triple sec"))
        self.limeJuiceLineEdit.setToolTip(_translate("MatrixWin", "Jiggers of lime juice"))
        self.limeJuiceLineEdit.setText(_translate("MatrixWin", "12.0"))
        self.iceHorizontalSlider.setToolTip(_translate("MatrixWin", "Chunks of ice"))
        self.label_6.setText(_translate("MatrixWin", "升"))
        self.label_3.setText(_translate("MatrixWin", "升"))
        self.label_8.setText(_translate("MatrixWin", "升"))
        self.label_5.setText(_translate("MatrixWin", "个"))

