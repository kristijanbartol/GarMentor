from enum import Enum


KPT_COLORS = {
    'red': (255, 0, 0),                 # 0
    'lime': (0, 255, 0),                # 1
    'blue': (0, 0, 255),                # 2
    'yellow': (255, 255, 0),            # 3
    'cyan': (0, 255, 255),              # 4
    'magenta': (255, 0, 255),           # 5
    'silver': (192, 192, 192),          # 6
    'maroon': (128, 0, 0),              # 7
    'green': (0, 128, 0),               # 8
    'purple': (128, 0, 128),            # 9
    'wheat': (245, 222, 179),           # 10
    'deeppink': (255, 20, 147),         # 11
    'white': (255, 255, 255),           # 12
    'indigo': (75, 0, 130),             # 13
    'midnightblue': (25, 25, 112),      # 14
    'lightskyblue': (135, 206, 250),    # 15
    'orange': (255, 165, 0)             # 16
}

LCOLOR = KPT_COLORS['red']
RCOLOR = KPT_COLORS['blue']


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


def norm_color(color_value):
    return tuple(x / 255. for x in color_value)


if __name__ == '__main__':
    for body_color in BodyColors:
        print(body_color)
