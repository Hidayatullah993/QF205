import sys1
from PyQt5.QtWidgets import QApplication, QWidget

class Test(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
    
    def initUI(self):
        self.resize(250,150)
        self.move(300,300)
        self.setWindowTitle("Test")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    test = Test()
    test.show()
    sys.exit(app.exec_())
