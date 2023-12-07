import time

from pushupModel import PushupEnsembleModel


class PushupCountMachine:
    def __init__(self):
        self.model = PushupEnsembleModel()
        self.prev_status = None
        self.curr_status = None
        self.cnt = 0
        self.status0 = [1, 4]
        self.status1 = [0, 5]
        self.status_abnormal = [2, 3]
        self.last_time = time.time()

    def count(self, data):
        self.prev_status = self.curr_status
        result = self.model.predict(data)
        if result is None:
            return self.cnt
        print("curr status : ", result[0])
        self.curr_status = result[0]

        if self.curr_status in self.status0 and self.prev_status in self.status1:
            curr_time = time.time()
            if curr_time - self.last_time >= 0.5:
                self.cnt += 1
                self.last_time = curr_time
        return self.cnt
