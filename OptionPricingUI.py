# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'C:\Users\ragha\Documents\SMU\SMU\Y3S1\QF205\QTProject\OptionPricing.ui'
#
# Created by: PyQt5 UI code generator 5.11.3
#
# WARNING! All changes made in this file will be lost!


#removed pandas as we didn't use it
from PyQt5 import QtCore, QtGui, QtWidgets
from options_calculator_with_greeks import get_all_values


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1079, 789)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(230, 30, 690, 61))
        font = QtGui.QFont()
        font.setFamily("Rockwell")
        font.setPointSize(24)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox.setGeometry(QtCore.QRect(30, 100, 451, 561))
        font = QtGui.QFont()
        font.setFamily("Century Gothic")
        font.setPointSize(16)
        self.groupBox.setFont(font)
        self.groupBox.setObjectName("groupBox")
        self.lineEdit_StockPrice = QtWidgets.QLineEdit(self.groupBox)
        self.lineEdit_StockPrice.setGeometry(QtCore.QRect(250, 140, 113, 30))
        self.lineEdit_StockPrice.setObjectName("lineEdit_StockPrice")
        self.lineEdit_ExercisePrice = QtWidgets.QLineEdit(self.groupBox)
        self.lineEdit_ExercisePrice.setGeometry(QtCore.QRect(250, 190, 113, 30))
        self.lineEdit_ExercisePrice.setObjectName("lineEdit_ExercisePrice")
        self.lineEdit_Volatility = QtWidgets.QLineEdit(self.groupBox)
        self.lineEdit_Volatility.setGeometry(QtCore.QRect(250, 310, 113, 30))
        self.lineEdit_Volatility.setObjectName("lineEdit_Volatility")
        self.lineEdit_InterestRate = QtWidgets.QLineEdit(self.groupBox)
        self.lineEdit_InterestRate.setGeometry(QtCore.QRect(250, 350, 113, 30))
        self.lineEdit_InterestRate.setObjectName("lineEdit_InterestRate")
        self.lineEdit_Dividend = QtWidgets.QLineEdit(self.groupBox)
        self.lineEdit_Dividend.setGeometry(QtCore.QRect(250, 390, 113, 30))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.lineEdit_Dividend.setFont(font)
        self.lineEdit_Dividend.setReadOnly(True)
        self.lineEdit_Dividend.setObjectName("lineEdit_Dividend")
        self.lineEdit_YieldRate = QtWidgets.QLineEdit(self.groupBox)
        self.lineEdit_YieldRate.setGeometry(QtCore.QRect(250, 430, 113, 30))
        self.lineEdit_YieldRate.setObjectName("lineEdit_YieldRate")
        self.dateEdit_ValueDate = QtWidgets.QDateEdit(self.groupBox)
        self.dateEdit_ValueDate.setGeometry(QtCore.QRect(250, 230, 121, 30))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.dateEdit_ValueDate.setFont(font)
        self.dateEdit_ValueDate.setObjectName("dateEdit_ValueDate")
        self.dateEdit_ExpirationDate = QtWidgets.QDateEdit(self.groupBox)
        self.dateEdit_ExpirationDate.setGeometry(QtCore.QRect(250, 270, 121, 30))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.dateEdit_ExpirationDate.setFont(font)
        self.dateEdit_ExpirationDate.setObjectName("dateEdit_ExpirationDate")
        self.label_2 = QtWidgets.QLabel(self.groupBox)
        self.label_2.setGeometry(QtCore.QRect(70, 150, 101, 20))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(self.groupBox)
        self.label_3.setGeometry(QtCore.QRect(70, 190, 111, 20))
        font = QtGui.QFont()
        font.setFamily("Century Gothic")
        font.setPointSize(10)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        self.label_4 = QtWidgets.QLabel(self.groupBox)
        self.label_4.setGeometry(QtCore.QRect(70, 230, 111, 20))
        font = QtGui.QFont()
        font.setFamily("Century Gothic")
        font.setPointSize(10)
        self.label_4.setFont(font)
        self.label_4.setObjectName("label_4")
        self.label_5 = QtWidgets.QLabel(self.groupBox)
        self.label_5.setGeometry(QtCore.QRect(70, 270, 121, 20))
        font = QtGui.QFont()
        font.setFamily("Century Gothic")
        font.setPointSize(10)
        self.label_5.setFont(font)
        self.label_5.setObjectName("label_5")
        self.label_6 = QtWidgets.QLabel(self.groupBox)
        self.label_6.setGeometry(QtCore.QRect(70, 320, 121, 20))
        font = QtGui.QFont()
        font.setFamily("Century Gothic")
        font.setPointSize(10)
        self.label_6.setFont(font)
        self.label_6.setObjectName("label_6")
        self.label_7 = QtWidgets.QLabel(self.groupBox)
        self.label_7.setGeometry(QtCore.QRect(70, 360, 151, 20))
        font = QtGui.QFont()
        font.setFamily("Century Gothic")
        font.setPointSize(10)
        self.label_7.setFont(font)
        self.label_7.setObjectName("label_7")
        self.label_8 = QtWidgets.QLabel(self.groupBox)
        self.label_8.setGeometry(QtCore.QRect(70, 390, 151, 20))
        font = QtGui.QFont()
        font.setFamily("Century Gothic")
        font.setPointSize(10)
        self.label_8.setFont(font)
        self.label_8.setObjectName("label_8")
        self.label_9 = QtWidgets.QLabel(self.groupBox)
        self.label_9.setGeometry(QtCore.QRect(70, 430, 121, 20))
        font = QtGui.QFont()
        font.setFamily("Century Gothic")
        font.setPointSize(10)
        self.label_9.setFont(font)
        self.label_9.setObjectName("label_9")
        self.pushButton_Calculate = QtWidgets.QPushButton(self.groupBox)
        self.pushButton_Calculate.setGeometry(QtCore.QRect(100, 480, 191, 51))
        font = QtGui.QFont()
        font.setBold(True)
        font.setItalic(True)
        font.setWeight(75)
        self.pushButton_Calculate.setFont(font)
        self.pushButton_Calculate.setObjectName("pushButton_Calculate")
        self.groupBox_2 = QtWidgets.QGroupBox(self.groupBox)
        self.groupBox_2.setGeometry(QtCore.QRect(10, 50, 411, 71))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.groupBox_2.setFont(font)
        self.groupBox_2.setObjectName("groupBox_2")
        self.radioButton_CrankN = QtWidgets.QRadioButton(self.groupBox_2)
        self.radioButton_CrankN.setGeometry(QtCore.QRect(230, 30, 161, 20))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.radioButton_CrankN.setFont(font)
        self.radioButton_CrankN.setObjectName("radioButton_CrankN")
        self.radioButton_Explicit = QtWidgets.QRadioButton(self.groupBox_2)
        self.radioButton_Explicit.setGeometry(QtCore.QRect(120, 30, 90, 20))
        font = QtGui.QFont()
        font.setFamily("Century Gothic")
        font.setPointSize(10)
        self.radioButton_Explicit.setFont(font)
        self.radioButton_Explicit.setObjectName("radioButton_Explicit")
        self.radioButton_Implicit = QtWidgets.QRadioButton(self.groupBox_2)
        self.radioButton_Implicit.setGeometry(QtCore.QRect(10, 30, 95, 20))
        font = QtGui.QFont()
        font.setFamily("Century Gothic")
        font.setPointSize(10)
        self.radioButton_Implicit.setFont(font)
        self.radioButton_Implicit.setChecked(True)
        self.radioButton_Implicit.setObjectName("radioButton_Implicit")
        self.groupBox_3 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_3.setGeometry(QtCore.QRect(500, 100, 511, 561))
        font = QtGui.QFont()
        font.setFamily("Century Gothic")
        font.setPointSize(16)
        self.groupBox_3.setFont(font)
        self.groupBox_3.setObjectName("groupBox_3")
        self.lineEdit_TheoreticalValue_Call = QtWidgets.QLineEdit(self.groupBox_3)
        self.lineEdit_TheoreticalValue_Call.setGeometry(QtCore.QRect(200, 60, 113, 22))
        self.lineEdit_TheoreticalValue_Call.setObjectName("lineEdit_TheoreticalValue_Call")
        self.lineEdit_TheoreticalValue_Put = QtWidgets.QLineEdit(self.groupBox_3)
        self.lineEdit_TheoreticalValue_Put.setGeometry(QtCore.QRect(340, 60, 113, 22))
        self.lineEdit_TheoreticalValue_Put.setObjectName("lineEdit_TheoreticalValue_Put")
        self.lineEdit_Delta_Call = QtWidgets.QLineEdit(self.groupBox_3)
        self.lineEdit_Delta_Call.setGeometry(QtCore.QRect(200, 90, 113, 22))
        self.lineEdit_Delta_Call.setObjectName("lineEdit_Delta_Call")
        self.lineEdit_Delta_Put = QtWidgets.QLineEdit(self.groupBox_3)
        self.lineEdit_Delta_Put.setGeometry(QtCore.QRect(340, 90, 113, 22))
        self.lineEdit_Delta_Put.setObjectName("lineEdit_Delta_Put")
        self.lineEdit_Delta100_Call = QtWidgets.QLineEdit(self.groupBox_3)
        self.lineEdit_Delta100_Call.setGeometry(QtCore.QRect(200, 120, 113, 22))
        self.lineEdit_Delta100_Call.setObjectName("lineEdit_Delta100_Call")
        self.lineEdit_Delta100_Put = QtWidgets.QLineEdit(self.groupBox_3)
        self.lineEdit_Delta100_Put.setGeometry(QtCore.QRect(340, 120, 113, 22))
        self.lineEdit_Delta100_Put.setObjectName("lineEdit_Delta100_Put")
        self.lineEdit_Lambda_Call = QtWidgets.QLineEdit(self.groupBox_3)
        self.lineEdit_Lambda_Call.setGeometry(QtCore.QRect(200, 150, 113, 22))
        self.lineEdit_Lambda_Call.setObjectName("lineEdit_Lambda_Call")
        self.lineEdit_Lambda_Put = QtWidgets.QLineEdit(self.groupBox_3)
        self.lineEdit_Lambda_Put.setGeometry(QtCore.QRect(340, 150, 113, 22))
        self.lineEdit_Lambda_Put.setObjectName("lineEdit_Lambda_Put")
        self.lineEdit_Gamma_Call = QtWidgets.QLineEdit(self.groupBox_3)
        self.lineEdit_Gamma_Call.setGeometry(QtCore.QRect(200, 180, 113, 22))
        self.lineEdit_Gamma_Call.setObjectName("lineEdit_Gamma_Call")
        self.lineEdit_Gamma_Put = QtWidgets.QLineEdit(self.groupBox_3)
        self.lineEdit_Gamma_Put.setGeometry(QtCore.QRect(340, 180, 113, 22))
        self.lineEdit_Gamma_Put.setObjectName("lineEdit_Gamma_Put")
        self.lineEdit_Gamma1_Call = QtWidgets.QLineEdit(self.groupBox_3)
        self.lineEdit_Gamma1_Call.setGeometry(QtCore.QRect(200, 210, 113, 22))
        self.lineEdit_Gamma1_Call.setObjectName("lineEdit_Gamma1_Call")
        self.lineEdit_Gamma1_Put = QtWidgets.QLineEdit(self.groupBox_3)
        self.lineEdit_Gamma1_Put.setGeometry(QtCore.QRect(340, 210, 113, 22))
        self.lineEdit_Gamma1_Put.setObjectName("lineEdit_Gamma1_Put")
        self.lineEdit_Theta_Call = QtWidgets.QLineEdit(self.groupBox_3)
        self.lineEdit_Theta_Call.setGeometry(QtCore.QRect(200, 240, 113, 22))
        self.lineEdit_Theta_Call.setObjectName("lineEdit_Theta_Call")
        self.lineEdit_Theta_Put = QtWidgets.QLineEdit(self.groupBox_3)
        self.lineEdit_Theta_Put.setGeometry(QtCore.QRect(340, 240, 113, 22))
        self.lineEdit_Theta_Put.setObjectName("lineEdit_Theta_Put")
        self.lineEdit_Theta7_Call = QtWidgets.QLineEdit(self.groupBox_3)
        self.lineEdit_Theta7_Call.setGeometry(QtCore.QRect(200, 270, 113, 22))
        self.lineEdit_Theta7_Call.setObjectName("lineEdit_Theta7_Call")
        self.lineEdit_Theta7_Put = QtWidgets.QLineEdit(self.groupBox_3)
        self.lineEdit_Theta7_Put.setGeometry(QtCore.QRect(340, 270, 113, 22))
        self.lineEdit_Theta7_Put.setObjectName("lineEdit_Theta7_Put")
        self.lineEdit_Vega_Call = QtWidgets.QLineEdit(self.groupBox_3)
        self.lineEdit_Vega_Call.setGeometry(QtCore.QRect(200, 300, 113, 22))
        self.lineEdit_Vega_Call.setObjectName("lineEdit_Vega_Call")
        self.lineEdit_Vega_Put = QtWidgets.QLineEdit(self.groupBox_3)
        self.lineEdit_Vega_Put.setGeometry(QtCore.QRect(340, 300, 113, 22))
        self.lineEdit_Vega_Put.setObjectName("lineEdit_Vega_Put")
        self.lineEdit_Rho_Call = QtWidgets.QLineEdit(self.groupBox_3)
        self.lineEdit_Rho_Call.setGeometry(QtCore.QRect(200, 330, 113, 22))
        self.lineEdit_Rho_Call.setObjectName("lineEdit_Rho_Call")
        self.lineEdit_Rho_Put = QtWidgets.QLineEdit(self.groupBox_3)
        self.lineEdit_Rho_Put.setGeometry(QtCore.QRect(340, 330, 113, 22))
        self.lineEdit_Rho_Put.setObjectName("lineEdit_Rho_Put")
        self.lineEdit_Psi_Call = QtWidgets.QLineEdit(self.groupBox_3)
        self.lineEdit_Psi_Call.setGeometry(QtCore.QRect(200, 360, 113, 22))
        self.lineEdit_Psi_Call.setObjectName("lineEdit_Psi_Call")
        self.lineEdit_Psi_Put = QtWidgets.QLineEdit(self.groupBox_3)
        self.lineEdit_Psi_Put.setGeometry(QtCore.QRect(340, 360, 113, 22))
        self.lineEdit_Psi_Put.setObjectName("lineEdit_Psi_Put")
        self.lineEdit_StrikeSensitivity_Call = QtWidgets.QLineEdit(self.groupBox_3)
        self.lineEdit_StrikeSensitivity_Call.setGeometry(QtCore.QRect(200, 390, 113, 22))
        self.lineEdit_StrikeSensitivity_Call.setObjectName("lineEdit_StrikeSensitivity_Call")
        self.lineEdit_StrikeSensitivity_Put = QtWidgets.QLineEdit(self.groupBox_3)
        self.lineEdit_StrikeSensitivity_Put.setGeometry(QtCore.QRect(340, 390, 113, 22))
        self.lineEdit_StrikeSensitivity_Put.setObjectName("lineEdit_StrikeSensitivity_Put")
        self.lineEdit_IntrinsicValue_Call = QtWidgets.QLineEdit(self.groupBox_3)
        self.lineEdit_IntrinsicValue_Call.setGeometry(QtCore.QRect(200, 420, 113, 22))
        self.lineEdit_IntrinsicValue_Call.setObjectName("lineEdit_IntrinsicValue_Call")
        self.lineEdit_IntrinsicValue_Put = QtWidgets.QLineEdit(self.groupBox_3)
        self.lineEdit_IntrinsicValue_Put.setGeometry(QtCore.QRect(340, 420, 113, 22))
        self.lineEdit_IntrinsicValue_Put.setObjectName("lineEdit_IntrinsicValue_Put")
        self.lineEdit_TimeValue_Call = QtWidgets.QLineEdit(self.groupBox_3)
        self.lineEdit_TimeValue_Call.setGeometry(QtCore.QRect(200, 450, 113, 22))
        self.lineEdit_TimeValue_Call.setObjectName("lineEdit_TimeValue_Call")
        self.lineEdit_TimeValue_Put = QtWidgets.QLineEdit(self.groupBox_3)
        self.lineEdit_TimeValue_Put.setGeometry(QtCore.QRect(340, 450, 113, 22))
        self.lineEdit_TimeValue_Put.setObjectName("lineEdit_TimeValue_Put")
        self.lineEdit_ZeroVolatility_Call = QtWidgets.QLineEdit(self.groupBox_3)
        self.lineEdit_ZeroVolatility_Call.setGeometry(QtCore.QRect(200, 480, 113, 22))
        self.lineEdit_ZeroVolatility_Call.setObjectName("lineEdit_ZeroVolatility_Call")
        self.lineEdit_ZeroVolatility_Put = QtWidgets.QLineEdit(self.groupBox_3)
        self.lineEdit_ZeroVolatility_Put.setGeometry(QtCore.QRect(340, 480, 113, 22))
        self.lineEdit_ZeroVolatility_Put.setObjectName("lineEdit_ZeroVolatility_Put")
        self.label_10 = QtWidgets.QLabel(self.groupBox_3)
        self.label_10.setGeometry(QtCore.QRect(30, 60, 151, 20))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_10.setFont(font)
        self.label_10.setObjectName("label_10")
        self.label_11 = QtWidgets.QLabel(self.groupBox_3)
        self.label_11.setGeometry(QtCore.QRect(30, 90, 120, 20))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_11.setFont(font)
        self.label_11.setObjectName("label_11")
        self.label_12 = QtWidgets.QLabel(self.groupBox_3)
        self.label_12.setGeometry(QtCore.QRect(30, 120, 120, 20))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_12.setFont(font)
        self.label_12.setObjectName("label_12")
        self.label_13 = QtWidgets.QLabel(self.groupBox_3)
        self.label_13.setGeometry(QtCore.QRect(30, 150, 120, 20))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_13.setFont(font)
        self.label_13.setObjectName("label_13")
        self.label_14 = QtWidgets.QLabel(self.groupBox_3)
        self.label_14.setGeometry(QtCore.QRect(30, 180, 120, 20))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_14.setFont(font)
        self.label_14.setObjectName("label_14")
        self.label_15 = QtWidgets.QLabel(self.groupBox_3)
        self.label_15.setGeometry(QtCore.QRect(30, 210, 120, 20))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_15.setFont(font)
        self.label_15.setObjectName("label_15")
        self.label_16 = QtWidgets.QLabel(self.groupBox_3)
        self.label_16.setGeometry(QtCore.QRect(30, 240, 120, 20))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_16.setFont(font)
        self.label_16.setObjectName("label_16")
        self.label_17 = QtWidgets.QLabel(self.groupBox_3)
        self.label_17.setGeometry(QtCore.QRect(30, 270, 120, 20))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_17.setFont(font)
        self.label_17.setObjectName("label_17")
        self.label_18 = QtWidgets.QLabel(self.groupBox_3)
        self.label_18.setGeometry(QtCore.QRect(30, 300, 120, 20))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_18.setFont(font)
        self.label_18.setObjectName("label_18")
        self.label_19 = QtWidgets.QLabel(self.groupBox_3)
        self.label_19.setGeometry(QtCore.QRect(30, 330, 120, 20))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_19.setFont(font)
        self.label_19.setObjectName("label_19")
        self.label_20 = QtWidgets.QLabel(self.groupBox_3)
        self.label_20.setGeometry(QtCore.QRect(30, 360, 120, 20))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_20.setFont(font)
        self.label_20.setObjectName("label_20")
        self.label_21 = QtWidgets.QLabel(self.groupBox_3)
        self.label_21.setGeometry(QtCore.QRect(30, 390, 141, 20))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_21.setFont(font)
        self.label_21.setObjectName("label_21")
        self.label_22 = QtWidgets.QLabel(self.groupBox_3)
        self.label_22.setGeometry(QtCore.QRect(30, 420, 120, 20))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_22.setFont(font)
        self.label_22.setObjectName("label_22")
        self.label_23 = QtWidgets.QLabel(self.groupBox_3)
        self.label_23.setGeometry(QtCore.QRect(30, 450, 120, 21))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_23.setFont(font)
        self.label_23.setObjectName("label_23")
        self.label_24 = QtWidgets.QLabel(self.groupBox_3)
        self.label_24.setGeometry(QtCore.QRect(30, 480, 120, 21))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_24.setFont(font)
        self.label_24.setObjectName("label_24")
        self.label_25 = QtWidgets.QLabel(self.groupBox_3)
        self.label_25.setGeometry(QtCore.QRect(200, 30, 120, 16))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_25.setFont(font)
        self.label_25.setObjectName("label_25")
        self.label_26 = QtWidgets.QLabel(self.groupBox_3)
        self.label_26.setGeometry(QtCore.QRect(340, 30, 120, 16))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_26.setFont(font)
        self.label_26.setObjectName("label_26")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1079, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        self.pushButton_Calculate.clicked.connect(self.testMethod)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label.setText(_translate("MainWindow", "Option Price (and Greeks) Calculator"))
        self.groupBox.setTitle(_translate("MainWindow", "Enter values here"))
        self.lineEdit_Dividend.setText(_translate("MainWindow", "Continuous"))
        self.label_2.setText(_translate("MainWindow", "Stock Price"))
        self.label_3.setText(_translate("MainWindow", "Exercise Price"))
        self.label_4.setText(_translate("MainWindow", "Value Date"))
        self.label_5.setText(_translate("MainWindow", "Expiration Date"))
        self.label_6.setText(_translate("MainWindow", "Volatility (%)"))
        self.label_7.setText(_translate("MainWindow", "Interest Rate (%)"))
        self.label_8.setText(_translate("MainWindow", "Dividend Method"))
        self.label_9.setText(_translate("MainWindow", "Yield Rate (%)"))
        self.pushButton_Calculate.setText(_translate("MainWindow", "Calculate"))
        self.groupBox_2.setTitle(_translate("MainWindow", "Black Scholes"))
        self.radioButton_CrankN.setText(_translate("MainWindow", "Crank - Nicolson"))
        self.radioButton_Explicit.setText(_translate("MainWindow", "Explicit"))
        self.radioButton_Implicit.setText(_translate("MainWindow", "Implicit"))
        self.groupBox_3.setTitle(_translate("MainWindow", "Results"))
        self.label_10.setText(_translate("MainWindow", "Theoretical Value"))
        self.label_11.setText(_translate("MainWindow", "Delta"))
        self.label_12.setText(_translate("MainWindow", "Delta 100\'s"))
        self.label_13.setText(_translate("MainWindow", "Lambda (%)"))
        self.label_14.setText(_translate("MainWindow", "Gamma"))
        self.label_15.setText(_translate("MainWindow", "Gamma (1%)"))
        self.label_16.setText(_translate("MainWindow", "Theta"))
        self.label_17.setText(_translate("MainWindow", "Theta (7 days)"))
        self.label_18.setText(_translate("MainWindow", "Vega"))
        self.label_19.setText(_translate("MainWindow", "Rho"))
        self.label_20.setText(_translate("MainWindow", "Psi"))
        self.label_21.setText(_translate("MainWindow", "Strike Sensitivity"))
        self.label_22.setText(_translate("MainWindow", "Intrinsic Value"))
        self.label_23.setText(_translate("MainWindow", "Time Value"))
        self.label_24.setText(_translate("MainWindow", "Zero Volatility"))
        self.label_25.setText(_translate("MainWindow", "Call"))
        self.label_26.setText(_translate("MainWindow", "Put"))


    def testMethod(self):

        Stock = float(self.lineEdit_StockPrice.text())
        Exercise_Price = float(self.lineEdit_ExercisePrice.text())
        Interest_rate = float(self.lineEdit_InterestRate.text())
        Volatility = float(self.lineEdit_Volatility.text())
        Yield_rate = float(self.lineEdit_YieldRate.text())
        Expiration_date = self.dateEdit_ExpirationDate.date()
        Value_date = self.dateEdit_ValueDate.date()

        print('Expiration_date: ', Expiration_date)
        print('Value_date: ', Value_date)

        algorithm = ''

        if self.radioButton_CrankN.isChecked():
            algorithm = 'crank nicolson'
        elif self.radioButton_Explicit.isChecked():
            algorithm = 'explicit'
        else:
            algorithm = 'implicit'

        values = get_all_values(
            Stock,
            Exercise_Price,
            Interest_rate,
            Volatility,
            Yield_rate,
            Expiration_date,
            Value_date,
            algorithm
        )
        
        call_value = "{:10.4f}".format(values['call']['value'])
        call_delta = "{:10.4f}".format(values['call']['delta'])
        call_delta_100 = "{:10.4f}".format(values['call']['delta_100'])
        call_lambda = "{:10.4f}".format(values['call']['lambda'])
        call_gamma = "{:10.4f}".format(values['call']['gamma'])
        call_gamma_1 = "{:10.4f}".format(values['call']['gamma_1%'])
        call_theta = "{:10.4f}".format(values['call']['theta'])
        call_theta_7d = "{:10.4f}".format(values['call']['theta_7d'])
        call_vega = "{:10.4f}".format(values['call']['vega'])
        call_rho = "{:10.4f}".format(values['call']['rho'])
        call_psi = "{:10.4f}".format(values['call']['psi'])
        call_strike_sensitivity = "{:10.4f}".format(values['call']['strike_sensitivity'])
        call_intrinsic_value = "{:10.4f}".format(values['call']['intrinsic_value'])
        call_time_value = "{:10.4f}".format(values['call']['time_value'])
        call_zero_volatility = "{:10.4f}".format(values['call']['zero_volatility'])
        
        print('Call Value (As String): ', call_value)

        put_value = "{:10.4f}".format(values['put']['value'])
        put_delta = "{:10.4f}".format(values['put']['delta'])
        put_delta_100 = "{:10.4f}".format(values['put']['delta_100'])
        put_lambda = "{:10.4f}".format(values['put']['lambda'])
        put_gamma = "{:10.4f}".format(values['put']['gamma'])
        put_gamma_1 = "{:10.4f}".format(values['put']['gamma_1%'])
        put_theta = "{:10.4f}".format(values['put']['theta'])
        put_theta_7d = "{:10.4f}".format(values['put']['theta_7d'])
        put_vega = "{:10.4f}".format(values['put']['vega'])
        put_rho = "{:10.4f}".format(values['put']['rho'])
        put_psi = "{:10.4f}".format(values['put']['psi'])
        put_strike_sensitivity = "{:10.4f}".format(values['put']['strike_sensitivity'])
        put_intrinsic_value = "{:10.4f}".format(values['put']['intrinsic_value'])
        put_time_value = "{:10.4f}".format(values['put']['time_value'])
        put_zero_volatility = "{:10.4f}".format(values['put']['zero_volatility'])

        
        self.lineEdit_TheoreticalValue_Call.setText(call_value)
        self.lineEdit_TheoreticalValue_Put.setText(put_value)
        self.lineEdit_Delta_Call.setText(call_delta)
        self.lineEdit_Delta_Put.setText(put_delta)
        self.lineEdit_Delta100_Call.setText(call_delta_100)
        self.lineEdit_Delta100_Put.setText(put_delta_100)
        self.lineEdit_Lambda_Call.setText(call_lambda)
        self.lineEdit_Lambda_Put.setText(put_lambda)
        self.lineEdit_Gamma_Call.setText(call_gamma)
        self.lineEdit_Gamma_Put.setText(put_gamma)
        self.lineEdit_Gamma1_Call.setText(call_gamma_1)
        self.lineEdit_Gamma1_Put.setText(put_gamma_1)
        self.lineEdit_Theta_Call.setText(call_theta)
        self.lineEdit_Theta_Put.setText(put_theta)
        self.lineEdit_Theta7_Call.setText(call_theta_7d)
        self.lineEdit_Theta7_Put.setText(put_theta_7d)
        self.lineEdit_Vega_Call.setText(call_vega)
        self.lineEdit_Vega_Put.setText(put_vega)
        self.lineEdit_Rho_Call.setText(call_rho)
        self.lineEdit_Rho_Put.setText(put_rho)
        self.lineEdit_Psi_Call.setText(call_psi)
        self.lineEdit_Psi_Put.setText(put_psi)
        self.lineEdit_StrikeSensitivity_Call.setText(call_strike_sensitivity)
        self.lineEdit_StrikeSensitivity_Put.setText(put_strike_sensitivity)
        self.lineEdit_IntrinsicValue_Call.setText(call_intrinsic_value)
        self.lineEdit_IntrinsicValue_Put.setText(put_intrinsic_value)
        self.lineEdit_TimeValue_Call.setText(call_time_value)
        self.lineEdit_TimeValue_Put.setText(put_time_value)
        self.lineEdit_ZeroVolatility_Call.setText(call_zero_volatility)
        self.lineEdit_ZeroVolatility_Put.setText(put_zero_volatility)
        

    

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

