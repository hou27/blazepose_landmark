from model import EnsembleModel


class CountMachine:
    def __init__(self):
        self.model = EnsembleModel()
        self.prev_status = None
        self.curr_status = None
        self.cnt = 0
        self.status0 = [1, 4]
        self.status1 = [0, 5]
        self.status_abnormal = [2, 3]

    def count(self, data):
        self.prev_status = self.curr_status
        self.curr_status = self.model.predict(data)[0]

        if self.curr_status in self.status0 and self.prev_status in self.status1:
            self.cnt += 1
        return self.cnt
