import csv

from PyQt5 import uic
from PyQt5.QtWidgets import QMainWindow
from keywords_compliance_checker import check_keywords_compliance


class MainWindow(QMainWindow):
    def __init__(self):
        QMainWindow.__init__(self)
        uic.loadUi('MainWindow.ui', self)

        self.push_check.clicked.connect(
            lambda: self._check(text=self.input_text.text(),
                                keywords=self.keywords.text()))

    def _check(self, text, keywords):
        keywords_arr = keywords.split()
        check_res, metric = check_keywords_compliance(text, keywords_arr)
        self.label_check.setText(str(check_res))
        self.metric.setText(str(metric))
        if metric:
            with open('metrics.csv', 'r+', newline='') as csvfile:
                fieldnames = ['keyword', 'metric']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                writer.writeheader()
                for keyword in keywords_arr:
                    writer.writerow({'keyword': keyword, 'metric': metric})
        self.show()
