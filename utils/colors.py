

from enum import Enum


class Colors(Enum):
    pass


class NoColors(Colors):
    
    WHITE  = (1., 1., 1.)
    BLACK  = (0., 0., 0.)


class GarmentColors(Colors):
    
    BLACK  = NoColors.BLACK
    GRAY   = (169., 169., 169.) / 255.
    GREEN  = (127., 255., 0.) / 255.
    BLUE   = (30., 144., 255) / 255.
    RED    = (255., 48., 48.) / 255.
    YELLOW = (255., 215., 0.) / 255.
    VIOLET = (238., 130., 238.) / 255.
    BROWN  = (156., 102., 31.) / 255.
    ORANGE = (255., 127., 0.) / 255.
    CYAN   = (0., 139., 139.) / 255.
    PINK   = (205., 16., 118.) / 255.
    PURPLE = (153., 50., 204.) / 255.
    GOLDEN = (255., 185., 15.) / 255.
    
    TORQUOISE   = (0., 197., 205.) / 255.
    DARK_BLUE   = (0., 0., 139.) / 255.
    DARK_GREEN  = (0., 128., 0.) / 255.
    DARK_VIOLET = (139., 34., 82.) / 255.
    LIGHT_BROWN = (245., 222., 179.) / 255.
    LIGHT_PINK  = (255., 225., 255.) / 255.
    OLIVE_GREEN = (110., 139., 61.) / 255.
    DARK_GOLDER = (139., 101., 8.) / 255.
    

class BodyColors(Colors):
    
    WHITE_SKIN  = (255., 211., 155.) / 255.
    DARK_SKIN   = (40., 40., 40.) / 255.
    TANNED_SKIN = (139., 90., 43.) / 255.
    OTHER_SKIN1 = (255., 218., 185.) / 255.
    OTHER_SKIN2 = (156., 102., 31.) / 255.


def A(color_value):
    return color_value + (1.,)
