import json
from pathlib import Path
from typing import Dict

import click
import cv2
from tqdm import tqdm


def detect(img_path: str) -> Dict[str, int]:
    """Object detection function, according to the project description, to implement.

    Parameters
    ----------
    img_path : str
        Path to processed image.

    Returns
    -------
    Dict[str, int]
        Dictionary with quantity of each object.
    """
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)

    #TODO: Implement detection method.
    size0 = 4080
    size1 = 3072
    scale = 1

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # threshold dla fioletowego
    threshold_p = cv2.inRange(hsv, (160, 65, 50), (170, 200, 200))
    dilated_img_p = cv2.dilate(threshold_p, (50, 50), iterations=2)
    (cnt_p, hier_p) = cv2.findContours(dilated_img_p, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    p = 0
    if (img.shape[0] != size0 & img.shape[1] != size0):
        if (img.shape[0] > img.shape[1]):
            scale = img.shape[0] / size0
        else:
            scale = img.shape[1] / size0

    for i in range(len(cnt_p)):
        if (cv2.contourArea(cnt_p[i]) > (400 * scale)):
            p = p + 1

    # threshold dla zielonego
    threshold_g = cv2.inRange(hsv, (36,110,110), (50, 250,240))
    dilated_img_g = cv2.erode(threshold_g, (20, 20), iterations=3)
    (cnt_g, hier_g) = cv2.findContours(dilated_img_g, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    g = 0
    if (img.shape[0] != size0 & img.shape[1] != size0):
        if (img.shape[0] > img.shape[1]):
            scale = img.shape[0] / size0
        else:
            scale = img.shape[1] / size0

    for i in range(len(cnt_g)):
        if (cv2.contourArea(cnt_g[i]) > (200 * scale)):
            g = g + 1

    # threshold dla zoltego
    threshold_y = cv2.inRange(hsv, (22,180,180), (30, 255,250))
    dilated_img_y = cv2.erode(threshold_y, (20, 20), iterations=3)
    (cnt_y, hier_y) = cv2.findContours(dilated_img_y, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    y = 0
    if (img.shape[0] != size0 & img.shape[1] != size0):
        if (img.shape[0] > img.shape[1]):
            scale = img.shape[0] / size0
        else:
            scale = img.shape[1] / size0

    for i in range(len(cnt_y)):
        if (cv2.contourArea(cnt_y[i]) > (200 * scale)):
            y = y + 1

    #threshold dla czerwonego
    threshold1 = cv2.inRange(hsv, (1, 65, 85), (8, 255, 255))
    threshold2 = cv2.inRange(hsv, (178, 65, 85), (180, 255, 255))
    threshold_r = threshold2 + threshold1
    dilated_img_r = cv2.dilate(threshold_r, (50, 50), iterations=2)

    (cnt_r, hier_r) = cv2.findContours(dilated_img_r, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    r = 0
    if (img.shape[0] != size0 & img.shape[1] != size0):
        if (img.shape[0] > img.shape[1]):
            scale = img.shape[0] / size0
        else:
            scale = img.shape[1] / size0

    for i in range(len(cnt_r)):
        if (cv2.contourArea(cnt_r[i]) > (700 * scale)):
            r = r + 1

    red = r
    yellow = y
    green = g
    purple = p

    return {'red': red, 'yellow': yellow, 'green': green, 'purple': purple}


@click.command()
@click.option('-p', '--data_path', help='Path to data directory', type=click.Path(exists=True, file_okay=False, path_type=Path), required=True)
@click.option('-o', '--output_file_path', help='Path to output file', type=click.Path(dir_okay=False, path_type=Path), required=True)
def main(data_path: Path, output_file_path: Path):
    img_list = data_path.glob('*.jpg')

    results = {}

    for img_path in tqdm(sorted(img_list)):
        fruits = detect(str(img_path))
        results[img_path.name] = fruits

    with open(output_file_path, 'w') as ofp:
        json.dump(results, ofp)


if __name__ == '__main__':
    main()
