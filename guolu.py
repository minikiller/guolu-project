import cv2
import numpy as np
import imutils
import setting
import matplotlib.pyplot as plt
import traceback
from PIL import Image, ImageDraw, ImageFont

useVideo = False
# videoFile = "20211022102003117.mp4"
# videoFile = "192.168.1.64_01_20211026152332696.mp4"
videoFile = "./video/02.mp4"
# videoFile = "sheng.mp4"

videoUrl = "rtsp://admin:kalix123@192.168.0.64:554/Streaming/channels/101/"

# cap = cv2.VideoCapture("1634869796094749.mp4")
# cap = cv2.VideoCapture("20211022101637930.mp4")
# 标记刻度，key为序号，value为区间
scale = setting.getAxis()
# scale = {0: (15, 25), 1: (5, 15), 2: (-5, 5), 3: (-15, -5), 4: (-25, -15)}
# scale = {0: (15, 25), 1: (5, 15), 2: (-5, 5), 3: (-15, -5), 4: (-25, -15)}
# 1 is red ,0 is green
color = setting.getColor()
# red color
# h_min = 0
# h_max = 10
# s_min = 43
# s_max = 255
# v_min = 46
# v_max = 255

# green
h_min = 35
h_max = 77
s_min = 43
s_max = 255
v_min = 46
v_max = 255

red_lower = np.array([h_min, s_min, v_min])
red_upper = np.array([h_max, s_max, v_max])


def cv2ImgAddText(img, text, left, top, textColor=(0, 255, 0), textSize=20):
    if (isinstance(img, np.ndarray)):  # 判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # 创建一个可以在给定图像上绘图的对象
    draw = ImageDraw.Draw(img)
    # 字体的格式
    fontStyle = ImageFont.truetype(
        "SimSun.ttf", textSize, encoding="utf-8")
    # 绘制文本
    draw.text((left, top), text, textColor, font=fontStyle)
    # 转换回OpenCV格式
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


def drawLine(red_bottom, red_left, green_top, green_right, img):
    # 起点和终点的坐标
    ptStart = (max(red_bottom), max(red_left))
    ptEnd = (min(green_top), min(green_right))
    print(ptStart)
    print(ptEnd)
    point_color = (0, 0, 255)  # BGR
    thickness = 2
    lineType = 4
    cv2.line(img, ptStart, ptEnd, point_color, thickness, lineType)


def detect():
    # cap = cv2.VideoCapture(0)

    # 使用网络摄像机
    # cap = cv2.VideoCapture()
    # cap.open("rtsp://admin:kalix123@192.168.0.64:554/Streaming/channels/101/")
    if useVideo:
        cap = cv2.VideoCapture()
        cap.open(videoUrl)
    else:
        cap = cv2.VideoCapture(videoFile)
    # cap = cv2.VideoCapture("1634869796094749.mp4")
    # cap = cv2.VideoCapture("20211022101637930.mp4")

    # Define the codec and create VideoWriter object
    fps = cap.get(cv2.CAP_PROP_FPS)  # fps
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(fps, width, height)
    wait_key = 30
    # out = cv2.VideoWriter('outvideo.mp4', cv2.VideoWriter_fourcc(
    #     *'DIVX'), fps, (width, height))
    try:
        while(True):
            ret, frame = cap.read()
            # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...22222")
                break
            values = setting.getScale()
            # process(frame, values)
            value, objs = findFive(frame, values)
            # for obj in objs:
            #     # [x1, y1, w, h] = cv2.boundingRect(obj)
            #     # x, y = (x1, y1), (x1+w-10, y1+h-10)
            #     # cv2.rectangle(frame, x, y, (255, 0, 0), 3)
            #     print('obj is ',obj)

            # guolu.hongyan(frame)
            # describe the type of font
            # to be used.
            font = cv2.FONT_HERSHEY_SIMPLEX

            # Use putText() method for
            # inserting text on video
            # cv2ImgAddText(frame,
            #               '中文: '+str(value),
            #               50, 50,
            #               (0, 0, 255),
            #               20)
            cv2.putText(frame,
                        'Values: '+str(value),
                        (50, 50),
                        font, 1,
                        (0, 0, 255),
                        2,
                        cv2.LINE_4)
            # out.write(frame)
            cv2.imshow("frame", frame)
            # 图片翻转
            # frame = cv2.flip(frame, 0)
            k = cv2.waitKey(wait_key) & 0xff
            if chr(k) == 'r':  # start running 按r键继续运行
                wait_key = 30
            elif chr(k) == 'p':  # pause between frames 按p键暂停
                wait_key = 0
            elif k == 27:  # end processing # 按esc结束
                break
            else:
                k = 0
    except Exception as e:
        print(e)
        print("An exception occurred")
        print(traceback.format_exc())
    finally:
        cap.release()
        # out.release()
        cv2.destroyAllWindows()

# TODO add logs
# TODO write data to serial number


def test():
    path = 'guolu01/guolu576.png'
    values = setting.getScale()
    img = cv2.imread(path)
    process(img, values)
    while(True):
        cv2.imshow("frame", img)
        # 图片翻转
        # frame = cv2.flip(frame, 0)
        k = cv2.waitKey(30) & 0xff
        if k == ord('q'):
            break
    cv2.destroyAllWindows()
    # x, y = (86, 45), (266, 585)
    # 85 42 184 542
    # path = 'image/guolu248.png'


"""
形状排序
"""


def sort_contours(cnts, method="left-to-right"):
    # initialize the reverse flag and sort index
    reverse = False
    i = 0

    # handle if we need to sort in reverse
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True

    # handle if we are sorting against the y-coordinate rather than
    # the x-coordinate of the bounding box
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1

    # construct the list of bounding boxes and sort them from top to
    # bottom
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                        key=lambda b: b[1][i], reverse=reverse))

    # return the list of sorted contours and bounding boxes
    return cnts, boundingBoxes


"""
根据颜色查看是否为红色还是绿色，目前绿色获得比较可靠
"""


def getColor(img):

    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # 转换空间
    red_mask = cv2.inRange(imgHSV, red_lower, red_upper)
    contours, _ = cv2.findContours(red_mask.copy(),
                                   cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if(len(contours) > 0):
        return 0
    else:
        return 1


"""
计算颜色,# 1 is red ,0 is green
"""


def calculateColor(objs):

    color1 = getColor(objs[0])
    color2 = getColor(objs[1])
    color3 = getColor(objs[2])
    color4 = getColor(objs[3])
    color5 = getColor(objs[4])
    return color.get((color1, color2, color3, color4, color5), -100)


"""
寻找五个矩形
"""


def calculateScale(objs, image):

    # 计算具体显示的数值
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    height = []
    cur_Value = 0

    for index, image in enumerate(objs):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        # show(thresh)
        opening = cv2.morphologyEx(
            thresh, cv2.MORPH_OPEN, kernel, iterations=1)
        cnts, _ = cv2.findContours(
            opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # cnts = imutils.grab_contours(cnts)
        # cv2.imshow("current"+str(index), opening)
        if len(cnts) > 1:
            print("this index is inside:", index)
            # area_list = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

            # print("length is :",len(area_list))
            # red_boxes = [cv2.boundingRect(c) for c in area_list]
            # font = cv2.FONT_HERSHEY_SIMPLEX
            (cnts, boundingBoxes) = sort_contours(cnts, method="top-to-bottom")
            for i, red_area in enumerate(cnts):
                area = cv2.contourArea(red_area)
                # print("current area:",area)
                if (area > 10):
                    x, y, w, h = cv2.boundingRect(red_area)
                    # print("red rectangle:",x, y, w, h)
                    height.append((x, y, w, h))
                    # result_img=cv2.rectangle(img, (x+range_x[0], y+range_x[1]), (x+w+range_x[0], y+h+range_x[1]), (255, 0,0 ), 1)
                    # result_img=cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0,0 ), 1)
                    # show(result_img)
            if len(height) == 2:  # 此时应该为红绿分割的图形，只计算分割点的坐标即可
                x1, y1, w1, h1 = height[0]
                x2, y2, w2, h2 = height[1]
                value = int((y2-(y1+h1))/2)  # 获得中间线的坐标
                current_height = h2+value
                total_height = y2+h2-y1
                min, max = scale[index]
                # print(min,max)
                cur_Value = min+current_height/(total_height/(max-min))

                # cv2.imshow("hello",image)
                print("current value:", cur_Value)
            else:
                print("current length is not in :", len(height))
            break
        else:
            print("this index is not inside:", index)
            continue
    if cur_Value == 0:  # 此时五个矩形应该是全红或者全绿状态。需要另外计算
        cur_Value = calculateColor(objs)
    return format(cur_Value, '.2f')
    # font = cv2.FONT_HERSHEY_SIMPLEX
    # cv2.putText(image, "current value is :"+str(cur_Value), (50, 50),
    #             font, 1.2, (0, 255, 255), 2, cv2.LINE_4)
    # if imgzi:
    #     cv2.imshow("hello", imgzi)

    # for img in objs:
    # cv2.imshow("color", np.hstack(
    #     [objs[0], objs[1], objs[2], objs[3], objs[4], ]))
    # 画1行5列图标大小5*5
    # L = 5

    # plt.subplots(1, L, figsize=(5, 5))
    # for i in range(L):
    #     #     print("current value:",i)
    #     plt.xticks(fontsize=10)  # 字体
    #     plt.title(str(i))  # 温度计编号

    #     # 模拟温度横线，画出来
    #     # plt.plot(3, 10, color='red', linewidth=0.5, linestyle='--')
    #     # 画出5个温度计每个间隔1位置
    #     plt.subplot(1, L, i+1)
    #     plt.imshow(imutils.opencv2matplotlib(objs[i]))

    # plt.show()


"""
寻找五个矩形
"""


def findFive(img, values):

    x1, y1, w, h = values
    x, y = (x1, y1), (x1+w-10, y1+h-10)
    cv2.rectangle(img, x, y, (255, 0, 0), 3)
    old_image = img
    ori_image = img.copy()
    img = img[x[1]:y[1], x[0]:y[0]]

    img = cv2.blur(img, (5, 5))
    # cv2.imshow("blur", img)
    image = img.copy()
    # ret2 , thresh = cv2.threshold(image,127,255,cv2.THRESH_BINARY_INV)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # 膨化操作，去除分割点
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.dilate(thresh, kernel, iterations=3)

    # 寻找形状
    contours, _ = cv2.findContours(
        thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    print("共计：", len(contours), "个形状")
    # 倒序后 要最大的前10个形状
    area_list = sorted(contours, key=cv2.contourArea, reverse=True)[:20]
    # 获得他们的坐标
    # red_boxes = [cv2.boundingRect(c) for c in area_list]
    (area_list, red_boxes) = sort_contours(area_list, method="top-to-bottom")

    # 最后保留的五个图形
    obj = []

    # 循环目标形状集合（最大的前10个）
    for i, red_area in enumerate(area_list):
        # 循环获得每个区域的面积
        area = cv2.contourArea(red_area)
        # 找出这个区域的坐标
        x, y, w, h = cv2.boundingRect(red_area)
        # 打印 最大的10个形状，找出面积数值 过滤掉噪声，留下5个温度计形状
        # print("current area:", area)  # 面积
        # print("red rectangle:x y w h", x, y, w, h)
        _x1, _y1 = (x1+x, y1+y), (x1+x + w, y1+y + h)
        # cv2.rectangle(img, x1, y1, (255, 0, 0), 1)
        # if ( w > 7 and h > 70):
        if (area > 20 and area < 4000 and w > 6 and w < 30 and h > 70):
            print("------------------------------")
            print("rectangle:x y w h", x, y, w, h)
            print("current area:", area)  # 面积
            cv2.rectangle(old_image, _x1, _y1, (255, 255, 255), 1)
            obj.append(img[y-5:y+h+5, x-5:x+w+5])  # 实体

            # result_img = cv2.rectangle(
            #     img, (x, y), (x + w, y + h), (36, 255, 12), 2)
            # cv2.imshow("five", result_img)
    # _, axs = plt.subplots(1, 5, figsize=(5, 5))
    # axs = axs.flatten()
    # for img, ax in zip(obj, axs):
    #     ax.imshow(img)
    # plt.draw()
    # plt.show()
    # plt.pause(0.01)

    if (obj != None and len(obj) == 5):
        value = calculateScale(obj, ori_image)
        return value, obj
    else:
        print("find no more index to 5!", len(obj))
        return 0, []


def process(img, values):
    # blur = cv2.medianBlur(img, 15)
    # x = (199, 105)
    # y = (346, 639)
    # x, y = (86, 45), (266, 585)
    # x, y = (86, 45), (266, 585)
    # x, y = (90, 53), (184, 542)
    x1, y1, w, h = values
    x, y = (x1, y1), (x1+w, y1+h)
    cv2.rectangle(img, x, y, (255, 0, 0), 3)
    img = img[x[1]:y[1], x[0]:y[0]]

    # img = cv2.GaussianBlur(img, (5, 5), 0)
    img = cv2.blur(img, (5, 5))
    cv2.imshow("blur", img)

    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # 转换空间

    h_min = 35
    h_max = 77
    s_min = 43
    s_max = 255
    v_min = 46
    v_max = 255
    print(h_min, h_max, s_min, s_max, v_min, v_max)
    green_lower = np.array([h_min, s_min, v_min])
    green_upper = np.array([h_max, s_max, v_max])
    green_mask = cv2.inRange(imgHSV, green_lower, green_upper)

    h_min = 0
    h_max = 10
    s_min = 43
    s_max = 255
    v_min = 46
    v_max = 255

    print(h_min, h_max, s_min, s_max, v_min, v_max)
    red_lower = np.array([h_min, s_min, v_min])
    red_upper = np.array([h_max, s_max, v_max])
    red_mask = cv2.inRange(imgHSV, red_lower, red_upper)

    contours, _ = cv2.findContours(red_mask.copy(),
                                   cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if(len(contours) > 0):
        area_list = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
        red_boxes = [cv2.boundingRect(c) for c in area_list]
        for red_area in area_list:
            area = cv2.contourArea(red_area)
            if (area > 100):
                x, y, w, h = cv2.boundingRect(red_area)
                result_img = cv2.rectangle(
                    img, (x, y), (x+w, y+h), (255, 0, 0), 1)
                cv2.imshow("red", result_img)

    contours, _ = cv2.findContours(green_mask.copy(),
                                   cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if(len(contours) > 0):
        area_list = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
        green_boxes = [cv2.boundingRect(c) for c in area_list]
        for red_area in area_list:
            area = cv2.contourArea(red_area)
            if (area > 100):
                x, y, w, h = cv2.boundingRect(red_area)
                result_img = cv2.rectangle(
                    img, (x, y), (x+w, y+h), (0, 255, 0), 1)
                cv2.imshow("green", result_img)

    # red_bottom = [c[0]+c[2] for c in red_boxes] # 红色的最下面的最右的一个点
    # green_top = [c[0] for c in green_boxes]  # 绿色的最上面的最左的一个点
    # red_left = [c[1]+c[3] for c in red_boxes]  # 红色的最下面的最右的一个点
    # green_right = [c[1] for c in green_boxes]  # 绿色的最上面的最左的一个点

    # drawLine(red_bottom, red_left, green_top, green_right, img)


"""
用来标记黑色面板的主要区域，并写入到配置文件(set.cfg)里面。
"""


def showLayout(image, i):

    # Load iamge, grayscale, adaptive threshold
    # image = cv2.imread('1.png')
    result = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 111, 9)
    # thresh = cv2.adaptiveThreshold(
    #     gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 51, 9)
    # thresh = cv2.adaptiveThreshold(
    # gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 9)

    # Fill rectangular contours
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        cv2.drawContours(thresh, [c], -1, (255, 255, 255), -1)

    # Morph open
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=7)
    # opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=5)

    # Draw rectangles
    cnts = cv2.findContours(opening, cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    # if len(cnts[0])>=2:
    #     print("current index:",i)
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        print("current index:", i, "value is:", x, y, w, h)
        cv2.rectangle(image, (x, y), (x + w, y + h), (36, 255, 12), 3)
    return (x, y, w, h)


def hongyan(image):
    im_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    im_gray = np.array(im_gray)
    # Step1.根据灰度图像中温度计区域的灰度阈值，大概进行二值化处理
    idx = np.where(im_gray < 70)
    img_bin = np.zeros_like(im_gray)
    img_bin[idx] = im_gray[idx].copy()
    img_bin[img_bin < 30] = 0

    # Step2. 去除过小的面积，根据温度计在图中宽高的大概尺寸，剔除小于这个大概尺寸的区域
    cnts = cv2.findContours(img_bin.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    # loop over the digit area candidates
    for c in cnts:
        # compute the bounding box of the contour
        (x, y, w, h) = cv2.boundingRect(c)
        if w < 50 or h < 50:  # 根据期望获取区域，即数字区域的实际高度预估28至50之间
            img_bin[y: y + h, x:x + w] = 0

    # Step3. 剔除温度计两侧的干扰，原理是统计统计每一行的宽度，根据前50行的宽，起始位置均值
    # 计算温度计的横坐标起始位置，以及温度计宽度
    r_idxs = []
    for i in range(img_bin.shape[0]):
        idx = np.array(np.where(img_bin[i, :] > 10))
        idx = idx[0]
        if len(idx) > 1 and (idx[-1] - idx[0] > 50):
            r_idxs.append([i, idx[-1] - idx[0], idx[0], idx[-1]])
    r_idxs = np.array(r_idxs)
    tmp1 = (r_idxs[:50, ].mean(axis=0))

    x1 = int(tmp1[2])
    x2 = int(tmp1[3])
    img_bin[:, :x1] = 0
    img_bin[:, x2:] = 0
    # Step4. 根据灰度中温度计的白色刻度大概定位高度的上下范围
    idx = np.where(im_gray > 140)
    img_bin[idx[0][-1] + 10:, ] = 0

    idx = np.where(img_bin > 10)

    y_start = idx[0][0]
    y_end = idx[0][-1]
    x_start = idx[1][0]
    x_end = idx[1][-1]
    cv2.rectangle(image, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)

    # cv2.imshow("img_bin", img_bin)
    # cv2.imshow("image", image)
    # # cv2.waitKey(0)
    # k = cv2.waitKey(30) & 0xff
    # if k == 27:
    #     break


def beginSetting():
    # cap = cv2.VideoCapture(0)

    # 使用网络摄像机
    # cap = cv2.VideoCapture()
    # cap.open("rtsp://admin:kalix123@192.168.0.64:554/Streaming/channels/101/")
    # cap = cv2.VideoCapture("guolu01.mp4")
    if useVideo:
        cap = cv2.VideoCapture()
        cap.open(videoUrl)
    else:
        cap = cv2.VideoCapture(videoFile)
    # cap = cv2.VideoCapture("20211022101637930.mp4")
    # cap = cv2.VideoCapture("1634869796094749.mp4")

    # Define the codec and create VideoWriter object
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(fps, width, height)
    # out = cv2.VideoWriter('basicvideo.mp4', cv2.VideoWriter_fourcc(*'DIVX'), fps, (width, height))
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (640,  480))
    i = 0
    try:
        while(True):
            ret, frame = cap.read()
            # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...22222")
                break
            i += 1
            result = showLayout(frame, i)
            print("get result", result)
            # guolu.hongyan(frame)
            # guolu.process(frame)
            # out.write(frame)
            cv2.imshow("frame", frame)
            # 图片翻转
            # frame = cv2.flip(frame, 0)
            k = cv2.waitKey(30) & 0xff
            if k == ord('s'):  # press s to save
                setting.writeScale(*result)
            elif k == ord('q'):  # press q to quit
                break
    except Exception as e:
        print(e)
        # print("An exception occurred")
    finally:
        cap.release()
        # out.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # beginSetting()
    detect()
    # test()
