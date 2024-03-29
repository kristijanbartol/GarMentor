from typing import (
    Dict, 
    List, 
    Union, 
    Optional
)
import random
import numpy as np


class GarmentClasses():

    """
    A garment classes management class.
    """

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
    VECTOR_SIZE = 4

    # NOTE (kbartol): Clothless currently not supported.
    CLOTHLESS_PROB = 0.0

    def _to_binary_vector(
            self, 
            upper_garment_class: Union[str, None], 
            lower_garment_class: Union[str, None]
        ) -> np.ndarray:
        """
        From upper and lower garment class string to binary labels vector.
        """
        binary_labels_vector: List[bool] = [False] * len(self.GARMENT_CLASSES)
        
        if upper_garment_class is not None:
            upper_label: int = self.UPPER_GARMENT_DICT[upper_garment_class]
            binary_labels_vector[upper_label] = True
        if lower_garment_class is not None:
            lower_label: int = self.LOWER_GARMENT_DICT[lower_garment_class]
            binary_labels_vector[lower_label] = True

        return np.array(binary_labels_vector, dtype=bool)

    def _generate_random_garment_classes(self) -> np.ndarray:
        """
        Generate random classes for upper for lower garment.
        """

        def get_random_garment_class(garment_classes: List[str]) -> str:
            '''Based on a random int in range, select the garment class.'''
            random_garment_int = random.randint(0, len(garment_classes) - 1)
            return garment_classes[random_garment_int]

        upper_garment_class = get_random_garment_class(
            self.UPPER_GARMENT_CLASSES
        )
        lower_garment_class = get_random_garment_class(
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

    def __init__(
            self, 
            upper_class: Optional[str] = None, 
            lower_class: Optional[str] = None
        ) -> None:
        """
        If nothing is already provided, initialize random classes.
        """
        if upper_class is None or lower_class is None:
            self.labels_vector = self._generate_random_garment_classes()
        else:
            self.labels_vector = self._to_binary_vector(upper_class, lower_class)

    @property
    def labels(self) -> Dict[str, Union[int, None]]:
        return {
            'upper': self.upper_label,
            'lower': self.lower_label
        }

    @property
    def upper_label(self) -> Union[int, None]:
        """
        Returns an int representing upper garment (see `GarmentClasses.GARMENT_DICT`).

        The method goes over the `GarmentClasses.UPPER_LABELS` and returns the index
        of the one where the `GarmentClasses.labels_vector` is 1. In an expected
        scenario, there should be only one such index. Note that label_list[0] means
        that the method should finally return a value instead of a list with a single
        element.
        """
        label_list = [x for x in self.UPPER_LABELS if self.labels_vector[x] == 1]
        if len(label_list) == 0:
            return None
        else:
            return label_list[0]

    @property
    def lower_label(self) -> Union[int, None]:
        """
        Returns an int representing lower garment (see `GarmentClasses.GARMENT_DICT`).

        The method goes over the `GarmentClasses.LOWER_LABELS` and returns the index
        of the one where the `GarmentClasses.labels_vector` is 1. In an expected
        scenario, there should be only one such index. Note that label_list[0] means
        that the method should finally return a value instead of a list with a single
        element.
        """
        label_list = [x for x in self.LOWER_LABELS if self.labels_vector[x] == 1]
        if len(label_list) == 0:
            return None
        else:
            return label_list[0]

    @property
    def classes(self) -> Dict[str, Union[str, None]]:
        """
        Returns a dictionary of upper and lower garment classes (as strings).
        """
        return {
            'upper': self.upper_class,
            'lower': self.lower_class
        }

    @property
    def upper_class(self) -> Union[str, None]:
        """
        Returns an upper garment class (see `GarmentClasses.GARMENT_CLASSES`).
        """
        if self.upper_label is None:
            return None
        else:
            return self.GARMENT_CLASSES[self.upper_label]

    @property
    def lower_class(self) -> Union[str, None]:
        """
        Returns a lower garment class (see `GarmentClasses.GARMENT_CLASSES`).
        """
        if self.lower_label is None:
            return None
        else:
            return self.GARMENT_CLASSES[self.lower_label]

    def __str__(self):
        """
        The string representation of the object ('{upper}+{lower}').
        """
        return f'{self.upper_class}+{self.lower_class}'

    def to_style_vector(
            self,
            upper_style: np.ndarray,
            lower_style: np.ndarray
    ) -> np.ndarray:
        """
        Create a style vector from the upper and lower style arrays.

        In particular, given two arrays, create a single 4x(vector_size) array
        which will contain upper and lower style values at the corresponding
        indices.
        """
        style_vector = np.zeros(
            shape=(self.NUM_CLASSES, self.VECTOR_SIZE),
            dtype=np.float32
        )
        style_vector[self.upper_label] = upper_style
        style_vector[self.lower_label] = lower_style

        return style_vector
