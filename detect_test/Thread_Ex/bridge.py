from threading import Lock

class Bridge:
    def __init__(self):
        self.counter = 0
        self.name = "아무개"
        self.address = "모름"
        self.lock = Lock()

    def across(self, name, address):
        self.lock.acquire()
        self.counter += 1
        self.name = name
        self.address = address
        self.check()
        self.lock.release()

    def toString(self):
        return "이름: {}, 출신:{}, 도전 횟수: {}".format(self.name, self.address, self.counter)
        

    def check(self):
        if self.name[0] != self.address[0]:
            print("문제 발생!!{}, {}".format(self.name[0], self.address[0]) + self.toString())