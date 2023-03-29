
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))



class ReticleArray():
    def __init__(self, **kwargs) -> None:
        self.coordinate = kwargs.get('coordinate', None)
        pass