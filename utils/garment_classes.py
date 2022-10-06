import random
from typing import List
import numpy as np


class GarmentClasses():

    '''A garment classes management class.'''

    # NOTE: Skirt is not used, for now.
    GARMENT_CLASSES = ['t-shirt', 'shirt', 'short-pant', 'pant']
    UPPER_GARMENT_CLASSES = ['t-shirt', 'shirt']
    LOWER_GARMENT_CLASSES = ['short-pant', 'pant']
    
    GARMENT_DICT = {
        't-shirt': 0,
        'shirt': 1,
        'short-pant': 2,
        'pant': 3
    }
    UPPER_GARMENT_DICT = {
        't-shirt': 0,
        'shirt': 1
    }
    LOWER_GARMENT_DICT = {
        'short-pant': 2,
        'pant': 3
    }

    UPPER_LABELS = [0, 1]
    LOWER_LABELS = [2, 3]

    NUM_CLASSES = len(GARMENT_CLASSES)

    CLOTHLESS_PROB = 0.05

    def _generate_random_garment_classes(self) -> np.ndarray:
        '''Generate random classes for upper for lower garment.'''

        def get_random_garment_class(garment_classes: List[str]) -> str:
            '''Based on a random int in range, select the garment class.'''
            random_garment_int = random.randint(0, len(garment_classes) - 1)
            return garment_classes[random_garment_int]

        upper_garment_class: str = get_random_garment_class(
            self.UPPER_GARMENT_CLASSES
        )
        lower_garment_class: str = get_random_garment_class(
            self.LOWER_GARMENT_CLASSES
        )

        random_number: float = random.uniform(0, 1)
        if random_number < self.CLOTHLESS_PROB:
            another_random_number = random.uniform(0, 1)
            if another_random_number < 0.5:
                upper_garment_class = None
            else:
                lower_garment_class = None

        binary_labels_vector: List[bool] = [0] * len(self.GARMENT_CLASSES)
        
        if upper_garment_class is not None:
            upper_label: int = self.UPPER_GARMENT_DICT[upper_garment_class]
            binary_labels_vector[upper_label]: bool = 1
        if lower_garment_class is not None:
            lower_label: int = self.LOWER_GARMENT_DICT[lower_garment_class]
            binary_labels_vector[lower_label]: bool = 1

        return np.array(binary_labels_vector, dtype=np.bool)

    def __init__(self, binary_labels_vector: np.ndarray = None):
        '''If nothing is already provided, initialize random classes.'''

        if binary_labels_vector is None:
            self.labels_vector: np.ndarray = self._generate_random_garment_classes()
        else:
            self.labels_vector: np.ndarray = binary_labels_vector

    @property
    def upper_label(self) -> int:
        '''Returns an int representing upper garment (see `GarmentClasses.GARMENT_DICT`).'''
        label_list = [x for x in self.UPPER_LABELS if self.labels_vector[x] == 1]
        if len(label_list) == 0:
            return None
        else:
            return label_list[0]

    @property
    def lower_label(self) -> int:
        '''Returns an int representing lower garment (see `GarmentClasses.GARMENT_DICT`).'''
        label_list = [x for x in self.LOWER_LABELS if self.labels_vector[x] == 1]
        if len(label_list) == 0:
            return None
        else:
            return label_list[0]

    @property
    def upper_class(self) -> str:
        '''Returns an upper garment class (see `GarmentClasses.GARMENT_CLASSES`).'''
        return self.GARMENT_CLASSES[self.upper_label]

    @property
    def lower_class(self) -> str:
        '''Returns a lower garment class (see `GarmentClasses.GARMENT_CLASSES`).'''
        return self.GARMENT_CLASSES[self.lower_label]
