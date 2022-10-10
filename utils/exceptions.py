
class GarmentorException(Exception):
    pass

class NakedGarmentorException(GarmentorException):
    
    def __init__(self):
        super().__init__('At least upper or lower garment should'
                         'not be None')