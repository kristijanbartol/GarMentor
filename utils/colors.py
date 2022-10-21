from enum import Enum


class Colors(Enum):
    pass


class NoColors(Colors):
    
    WHITE  = (255., 255., 255.)
    BLACK  = (0., 0., 0.)


class GarmentColors(Colors):
    
    BLACK  = (0., 0., 0.)
    GRAY   = (169., 169., 169.)
    GREEN  = (127., 255., 0.)
    BLUE   = (30., 144., 255)
    RED    = (255., 48., 48.)
    YELLOW = (255., 215., 0.)
    VIOLET = (238., 130., 238.)
    BROWN  = (156., 102., 31.)
    ORANGE = (255., 127., 0.)
    CYAN   = (0., 139., 139.)
    PINK   = (205., 16., 118.)
    PURPLE = (153., 50., 204.)
    GOLDEN = (255., 185., 15.)
    
    TORQUOISE   = (0., 197., 205.)
    DARK_BLUE   = (0., 0., 139.)
    DARK_GREEN  = (0., 128., 0.)
    DARK_VIOLET = (139., 34., 82.)
    LIGHT_BROWN = (245., 222., 179.)
    LIGHT_PINK  = (255., 225., 255.)
    OLIVE_GREEN = (110., 139., 61.)
    DARK_GOLDER = (139., 101., 8.)
    

class BodyColors(Colors):
    
    WHITE_SKIN  = (255., 211., 155.)
    DARK_SKIN   = (40., 40., 40.)
    TANNED_SKIN = (139., 90., 43.)
    OTHER_SKIN1 = (255., 218., 185.)
    OTHER_SKIN2 = (156., 102., 31.)


def N(color_value):
    return tuple(x / 255. for x in color_value)


def A(color_value):
    return color_value + (1.,)


if __name__ == '__main__':
    for body_color in BodyColors:
        print(body_color)
