import random


class GarmentClasses():

    GARMENT_CLASSES = ['t-shirt', 'shirt', 'short-pant', 'pant', 'skirt']
    UPPER_GARMENT_CLASSES = ['t-shirt', 'shirt']
    LOWER_GARMENT_CLASSES = ['short-pant', 'pant', 'skirt']
    
    GARMENT_DICT = {
        't-shirt': 0,
        'shirt': 1,
        'short-pant': 2,
        'pant': 3,
        'skirt': 4
    }
    UPPER_GARMENT_DICT = {
        't-shirt': 0,
        'shirt': 1
    }
    LOWER_GARMENT_DICT = {
        'short-pant': 2,
        'pant': 3,
        'skirt': 4
    }

    UPPER_LABELS = [0, 1]
    LOWER_LABELS = [2, 3, 4]

    NUM_CLASSES = len(GARMENT_CLASSES)

    # TODO: Set this > 0. to use.
    # TODO: Move this to configuration.
    CLOTHLESS_PROB = 0.

    def _generate_random_garment_classes(self):

        def get_random_garment_class(garment_classes):
            random_garment_int = random.randint(0, len(garment_classes) - 1)
            return garment_classes[random_garment_int]

        upper_garment_class = get_random_garment_class(self.UPPER_GARMENT_CLASSES)
        lower_garment_class = get_random_garment_class(self.LOWER_GARMENT_CLASSES)

        random_number = random.uniform(0, 1)
        if random_number < self.CLOTHLESS_PROB:
            another_random_number = random.uniform(0, 1)
            if another_random_number < 0.5:
                upper_garment_class = None
            else:
                lower_garment_class = None

        upper_label = self.UPPER_GARMENT_DICT[upper_garment_class]
        lower_label = self.LOWER_GARMENT_DICT[lower_garment_class]

        binary_label_vector = [0] * len(self.GARMENT_CLASSES)
        
        if upper_label is not None:
            binary_label_vector[upper_label] = 1
        if lower_label is not None:
            binary_label_vector[lower_label] = 1

        return binary_label_vector

    def __init__(self, binary_label_vector=None):
        if binary_label_vector is None:
            self.label_vector = self._generate_random_garment_classes()
        else:
            self.label_vector = binary_label_vector

    @property
    def upper_label(self):
        label_list = [x for x in self.UPPER_LABELS if self.label_vector[x] == 1]
        if len(label_list) == 0:
            return None
        else:
            return label_list[0]

    @property
    def lower_label(self):
        label_list = [x for x in self.LOWER_LABELS if self.label_vector[x] == 1]
        if len(label_list) == 0:
            return None
        else:
            return label_list[0]

    @property
    def upper_class(self):
        return self.GARMENT_CLASSES[self.upper_label]

    @property
    def lower_class(self):
        return self.GARMENT_CLASSES[self.lower_label]
