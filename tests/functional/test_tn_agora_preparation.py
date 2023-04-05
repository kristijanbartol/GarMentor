import cv2


if __name__ == '__main__':
    img = cv2.imread('/data/garmentor/agora/full_img/archviz/720x1280/ag_trainset_axyz_bfh_archviz_5_10_cam03_00023.png')
    img_crop = img[700:1000, 1200:1500]

    cv2.imwrite('/garmentor/tests/functional/out/original.png', img)
    cv2.imwrite('/garmentor/tests/functional/out/crop_test.png', img_crop)
