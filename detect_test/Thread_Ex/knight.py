from threading import Thread

class Knight(Thread):
    def __init__(self, bridge, name, address):
        super().__init__()
        self.bridge = bridge
        self.name = name
        self.address = address
    
    def run(self):
        print(self.name, "기사가 도전한다.")
        
        for i in range(100000):
            self.bridge.across(self.name, self.address)
        