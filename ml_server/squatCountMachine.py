from squatModel import SquatEnsembleModel


class SquatCountMachine:
    def __init__(self):
        self.model = SquatEnsembleModel()
        self.prev_status = None
        self.curr_status = None
        self.cnt = 0
        self.status0 = [0, 1, 2, 4]
        self.status1 = [3, 5]

    def count(self, data):
        self.prev_status = self.curr_status
        self.curr_status = self.model.predict(data)[0]

        if self.curr_status in self.status0 and self.prev_status in self.status1:
            self.cnt += 1
        return self.cnt
