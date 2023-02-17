import random
from typing import Dict, List
import numpy as np
import torch


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

    # NOTE (kbartol): Clothless currently not supported.
    CLOTHLESS_PROB = 0.0       

    def __init__(self, upper_class: str = None, lower_class: str = None):
        '''If nothing is already provided, initialize random classes.'''

        self.labels_vector: np.ndarray[bool] = None
        if upper_class is None or lower_class is None:
            self.labels_vector = self._generate_random_garment_classes()
        else:
            self.labels_vector = self._to_binary_vector(upper_class, lower_class)

    def _to_binary_vector(
            self, 
            upper_garment_class: str = None, 
            lower_garment_class: str = None
        ) -> np.ndarray[bool]:
        '''From garment classes strings to bool array of binary labels.'''
        binary_labels_vector: List[bool] = [0] * len(self.GARMENT_CLASSES)
        
        if upper_garment_class is not None:
            upper_label: int = self.UPPER_GARMENT_DICT[upper_garment_class]
            binary_labels_vector[upper_label]: bool = 1
        if lower_garment_class is not None:
            lower_label: int = self.LOWER_GARMENT_DICT[lower_garment_class]
            binary_labels_vector[lower_label]: bool = 1

        return np.array(binary_labels_vector, dtype=np.bool)

    def _generate_random_garment_classes(self) -> None:
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

        return self._to_binary_vector(upper_garment_class, lower_garment_class) 

    @property
    def labels(self) -> Dict[str, int]:
        return {
            'upper': self.upper_label,
            'lower': self.lower_label
        }

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
    def classes(self) -> Dict[str, str]:
        return {
            'upper': self.upper_class,
            'lower': self.lower_class
        }

    @property
    def upper_class(self) -> str:
        '''Returns an upper garment class (see `GarmentClasses.GARMENT_CLASSES`).'''
        if self.upper_label is None:
            return None
        else:
            return self.GARMENT_CLASSES[self.upper_label]

    @property
    def lower_class(self) -> str:
        '''Returns a lower garment class (see `GarmentClasses.GARMENT_CLASSES`).'''
        if self.lower_label is None:
            return None
        else:
            return self.GARMENT_CLASSES[self.lower_label]

    def __str__(self):
        return f'{self.upper_class}+{self.lower_class}'

    @staticmethod
    def to_binary_logits(labels_vector: torch.Tensor[bool]):
        upper_class_logit = 1. if labels_vector[1] == 1. else 0.
        lower_class_logit = 1. if labels_vector[3] == 1. else 0.
        return upper_class_logit, lower_class_logit

    @staticmethod
    def logits_to_labels_vector(
            upper_class_logit: float, 
            lower_class_logit: float
    ) -> torch.Tensor[bool]:
        upper_class_int = torch.round(upper_class_logit)
        lower_class_int = torch.round(lower_class_logit)

        upper_label = 1 if upper_class_int == 1 else 0
        lower_label = 3 if lower_class_int == 1 else 2
        
        labels_vector = [0] * len(GarmentClasses.GARMENT_CLASSES)
        labels_vector[upper_label] = 1
        labels_vector[lower_label] = 1

        return labels_vector
