from configparser import ConfigParser
import codecs

config = ConfigParser()

# Open the file with the correct encoding


def getScale():
    config.read('set.cfg')
    x = int(config.get('default', 'x'))
    y = int(config.get('default', 'y'))
    w = int(config.get('default', 'w'))
    h = int(config.get('default', 'h'))
    return (x, y, w, h)

# 获得坐标


def getAxis():
    config.read('set.cfg')
    # Get one section in a dict
    # tuple(map(int, str.split(','))) 把字符串分割成tuple
    # dict = {int(k): v for k, v in config['axis'].items()}
    dict = {int(k): tuple(map(int, v.split(',')))
            for k, v in config['axis'].items()}
    return dict

# 显示颜色的配置数值


def getColor():
    config.read('set.cfg')
    color0 = int(config.get('color', '0'))
    color1 = int(config.get('color', '1'))
    color2 = int(config.get('color', '2'))
    color3 = int(config.get('color', '3'))
    color4 = int(config.get('color', '4'))
    color5 = int(config.get('color', '5'))
    return {(0, 0, 0, 0, 0): color0,
     (1, 0, 0, 0, 0): color1,
     (1, 1, 0, 0, 0): color2,
     (1, 1, 1, 0, 0): color3,
     (1, 1, 1, 1, 0): color4,
     (1, 1, 1, 1, 1): color5}


def writeScale(*values):

    x, y, w, h = values
    config['default'] = {
        "x": x, "y": y,
        "w": w, "h": h
    }

    with open('set.cfg', 'w') as configfile:
        config.write(configfile)


if __name__ == "__main__":
    # mylist = [10, 20, 30, 40]
    # writeScale(*mylist)
    print(getColor())
    # print(getAxis())
