import os
import math
import cv2
import numpy as np

import torch
import torch.utils.data as data

DATA_PATH = "./data"  # defalut data path


def crop(src: np.ndarray, width_range: tuple, height_range: tuple) -> np.ndarray:
    """
    param
        src:            image to be cropped. (height, width, channels)
        width_range:    indicate cropping start and end index about width. (crop start index about width, crop end index about width)
        height_range:   indicate cropping start and end index about height. (crop start index about height, crop end index about height)
    return
        dst:            cropped Image. It has (heigth_range[1] - height_range[0], width_range[1] - width_range[0], channels) shape.
    """
    dst = src.copy()
    dst = src[height_range[0] : height_range[1], width_range[0] : width_range[1], :]

    return dst


def crop_img(src: np.ndarray, size: tuple, strides: int) -> np.ndarray:
    """
    param
        src:        image to be cropped. (heigth, width, channels)
        size:       cropped image size. (width, height)
        strides:    crop filter's stride.
    return
        crop_set:   cropped image set. (images num, height, width, channels)
    """
    src_h, src_w, src_c = src.shape

    crop_set = []

    for w_idx in range(0, src_w - size[0], strides):
        for h_idx in range(0, src_h - size[1], strides):
            crop_set.append(crop(src, (w_idx, w_idx + size[0]), (h_idx, h_idx + size[1])))
    crop_set = np.array(crop_set)

    return crop_set


def rotate(src: np.ndarray, angle: int) -> np.ndarray:
    """
    param
        src:    image to be rotated. (height, width, channels)
        angle:  rotation angle in degrees.
    return
        dst:    rotated Image. Rotated image has empty space. To remove this space, dst resized.
    """
    src_h, src_w, src_c = src.shape
    rotate_mat = cv2.getRotationMatrix2D((src_w / 2, src_h / 2), angle, 1)
    dst = cv2.warpAffine(src, rotate_mat, (src_w, src_h))

    removed_len_h = abs(math.ceil(math.tan(math.pi / 180 * angle) * src_w / 2))
    removed_len_w = abs(math.ceil(math.tan(math.pi / 180 * angle) * src_h / 2))

    dst = dst[removed_len_h:-removed_len_h, removed_len_w:-removed_len_w, :]

    return dst


def rotate_img(src: np.ndarray, angle: int, step: int, **optional) -> np.ndarray:
    """
    param
        src:            image to be rotated. (height, width, channels)
        angle:          rotation angle in degrees.
        step:           image would be rotated by increasing by step value from 0 degrees to angle variable.
        reverse_flag:   *optional* If this flag is True, add also image rotated by reverse angle.
    return
        dst:            rotated image set. (images num, height, width, channels)
    """
    src_h, src_w, src_c = src.shape
    min_h, min_w = src_h, src_w

    rotate_set = []

    for angle_i in range(step, step + angle, step):
        rotate_set.append(rotate(src, angle_i))
        if "reverse_flag" in optional.keys() and optional["reverse_flag"]:
            rotate_set.append(rotate(src, -angle_i))
        if min_h < rotate_set[-1].shape[0]:
            min_h = rotate_set[-1].shape[0]
        if min_w < rotate_set[-1].shape[1]:
            min_w = rotate_set[-1].shape[1]

    # each rotate images has different size, so make all images the size(minimum height, minimum width, channels) that the smallest image has.
    for i in range(len(rotate_set)):
        rotate_set[i] = cv2.resize(rotate_set[i], dsize=(min_w, min_h), interpolation=cv2.INTER_AREA)
    rotate_set = np.array(rotate_set)

    return rotate_set


def extract_red(src: np.ndarray) -> np.ndarray:
    """
    param
        src:        image to extract red color. (height, width, channels)
    return
        extracted:  image that consists of only red color of src. (height, width, channels)
    """
    hsv_img = cv2.cvtColor(src, cv2.COLOR_RGB2HSV)
    lower_red = cv2.inRange(hsv_img, (0, 100, 100), (10, 255, 255))
    upper_red = cv2.inRange(hsv_img, (170, 100, 100), (180, 255, 255))
    mask = cv2.addWeighted(lower_red, 1.0, upper_red, 1.0, 0.0)

    extracted = cv2.bitwise_and(hsv_img, hsv_img, mask=mask)
    extracted = cv2.cvtColor(extracted, cv2.COLOR_HSV2RGB)

    return extracted


class load_data(data.Dataset):
    def __init__(self, src_path, class_num):
        """
        pram
            src_path:   directory where has image data sets. This must have directories class_0, class_1, ..., class_{class_num - 1}
            class_num:  the num of data set's class.
        """
        super(load_data, self).__init__()

        self.file_list = []
        self.label_a = []

        for i in range(class_num):  # sanity check.
            assert os.path.isdir(os.path.join(src_path, f"{i}")), f"class_{i} does not exist in {src_path}"

        for i in range(class_num):
            image_list = os.listdir(os.path.join(src_path, f"{i}"))
            for image_name in image_list:
                self.file_list.append(os.path.join(src_path, f"{i}", image_name))
                self.label_a.append(i)

        self.file_list = np.array(self.file_list)
        self.label_a = torch.tensor(self.label_a, dtype=torch.long)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            img_set = []
            red_set = []
            for i in range(*idx.indices(len(self))):
                img_bgr = cv2.imread(self.file_list[i], cv2.IMREAD_COLOR)
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                img_set.append(img_rgb.transpose(2, 0, 1).astype(np.float32))
                red_set.append(extract_red(img_rgb).transpose(2, 0, 1).astype(np.float32))
            img_set = np.stack(img_set)
            img_set = torch.from_numpy(img_set)
            red_set = np.stack(red_set)
            red_set = torch.from_numpy(red_set)
            return img_set, red_set, self.label_a[idx]

        elif isinstance(idx, np.ndarray):
            img_set = []
            red_set = []
            for i in idx:
                img_bgr = cv2.imread(self.file_list[i], cv2.IMREAD_COLOR)
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                img_set.append(img_rgb.transpose(2, 0, 1).astype(np.float32))
                red_set.append(extract_red(img_rgb).transpose(2, 0, 1).astype(np.float32))
            img_set = np.stack(img_set)
            img_set = torch.from_numpy(img_set)
            red_set = np.stack(red_set)
            red_set = torch.from_numpy(red_set)
            return img_set, red_set, self.label_a[idx]

        else:
            img_bgr = cv2.imread(self.file_list[idx], cv2.IMREAD_COLOR)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            red_rgb = extract_red(img_rgb)
            img_rgb = torch.from_numpy(img_rgb.transpose(2, 0, 1).astype(np.float32))
            red_rgb = torch.from_numpy(red_rgb.transpose(2, 0, 1).astype(np.float32))
            return img_rgb, red_rgb, self.label_a[idx]

    def __len__(self):
        return len(self.label_a)


def inc_data(src_path: str, dst_path: str, class_num: list):
    """
    right label (label 1) has 8 images for each class
    """
    # rotate -> 4 per an image
    # crop -> when step is 0.12 : 4 per an image / when step is 0.06 : 9 per an image
    ROTATE_ANGLE = 20
    ROTATE_STEP = 10
    SIZE_SCALE = 0.88
    STEP_SCALE = 0.06

    CLASS_NUM = class_num

    assert os.path.isdir(dst_path), f"{dst_path} does not exist."

    for class_i in CLASS_NUM:  # make directory
        if not os.path.isdir(os.path.join(dst_path, f"{class_i}")):
            os.mkdir(os.path.join(dst_path, f"{class_i}"))

    for class_i in CLASS_NUM:
        target_dir = os.path.join(src_path, f"{class_i}")
        file_list = os.listdir(target_dir)
        for file_idx in range(len(file_list)):
            img = cv2.imread(os.path.join(target_dir, file_list[file_idx]), cv2.IMREAD_COLOR)
            # rotate
            rotate_imgs = rotate_img(img, ROTATE_ANGLE, ROTATE_STEP, reverse_flag=True)
            for rotate_idx in range(rotate_imgs.shape[0]):
                src_h, src_w, src_c = rotate_imgs[rotate_idx].shape  # src_h == src_w
                # crop
                crop_imgs = crop_img(rotate_imgs[rotate_idx], (int(src_w * SIZE_SCALE), int(src_h * SIZE_SCALE)), int(src_w * STEP_SCALE))
                for crop_idx in range(crop_imgs.shape[0]):
                    # resize
                    target_img = cv2.resize(crop_imgs[crop_idx], dsize=(256, 256), interpolation=cv2.INTER_AREA)
                    cv2.imwrite(os.path.join(dst_path, f"{class_i}", f"{class_i}_{file_idx}_{rotate_idx}_{crop_idx}.jpg"), target_img)
                    print(
                        f"[class {class_i}][file {file_idx + 1}/{len(file_list)}][rotate {rotate_idx + 1}/{rotate_imgs.shape[0]}][crop {crop_idx + 1}/{crop_imgs.shape[0]}]" + " " * 20,
                        end="\r",
                    )
        print(f"[class {class_i}] [ done ]" + " " * 50)


def progress_bar(epoch_idx: int, epoch_size: int, batch_idx: int, batch_size: int):
    cur_progress = int(50 * (batch_idx + 1) / batch_size)
    cur_str = "#" * cur_progress
    print(f"[epoch : {epoch_idx:03d}/{epoch_size:03d}] {cur_str:-<50}", end="\r")


if __name__ == "__main__":
    pass
    #inc_data(src_path=os.path.join(DATA_PATH, "source", "train"), dst_path=os.path.join(DATA_PATH, "train"), class_num=[0, 1, 2, 3])
    #inc_data(src_path=os.path.join(DATA_PATH, "source", "test"), dst_path=os.path.join(DATA_PATH, "test"), class_num=[0, 1, 2, 3])

    # test_data = load_data(src_path=os.path.join(DATA_PATH, "train"), class_num=4)
    # print(len(test_data))
