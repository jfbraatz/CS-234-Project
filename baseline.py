from warfarinbandit import WarfarinBandit

class Baseline1(WarfarinBandit):
    # Performance: 0.6146888567293777
    
    def predict(self, x):
        return 1

baseline = Baseline1()
baseline.simulate()