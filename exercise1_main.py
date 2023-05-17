import argparse
import os
import platform
import sys
from pathlib import Path
import glob
from tqdm import tqdm
import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from yolov5.models.common import DetectMultiBackend
from yolov5.utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from yolov5.utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from yolov5.utils.plots import Annotator, colors, save_one_box
from yolov5.utils.torch_utils import select_device, smart_inference_mode

from my_utils import helper

def run(
        weights=ROOT / 'yolov5/runs/train/yolov5s_results8/weights/best.pt',
        source=ROOT / 'yolov5/custom_test/test/images',
        reference_heroes=ROOT / 'crawled_images',
        data=ROOT / 'yolov5/data/coco128.yaml',  # dataset.yaml path
        imgsz=(416, 416),  # inference size (height, width)
        conf_thres=0.4,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='cpu',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        project=ROOT / 'runs/detect_hero',  # save results to project/name
        name='result_',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
):
    source = str(source)
    result_folder = helper.increment_path(Path(project) / name, exist_ok=exist_ok)
    report_name = "output.txt"
    report_file = helper.generate_treefolder(report_name, str(result_folder))

    # YoloV5
    device = select_device(device)
    image_paths_list = []
    if os.path.isfile(source):
        image_paths_list.append(source)
    elif os.path.isdir(source):
        image_paths_list = glob.glob(source+"/*.png", recursive=True) + glob.glob(source+"/*.jpg", recursive=True) 

    wild_rift_heroes = glob.glob(str(reference_heroes)+"/*.png", recursive=True) + glob.glob(str(reference_heroes)+"/*.png", recursive=True)
    print(image_paths_list)

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    model.warmup(imgsz=(1, 3, *imgsz))

    output_result = []

    # Load data
    with tqdm(total=len(image_paths_list)) as progress_bar:
        for image_path in image_paths_list:
            dataset = LoadImages(image_path, img_size=imgsz, stride=stride,
                                 auto=pt, vid_stride=1)
            for path, im, im0s, vid_cap, s in dataset:
                im = torch.from_numpy(im).to(model.device)
                im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
                im /= 255  # 0 - 255 to 0.0 - 1.0
                if len(im.shape) == 3:
                    im = im[None]  # expand for batch dim

                # Inference
                pred = model(im, augment=False, visualize=False)

                # NMS
                pred = non_max_suppression(pred, conf_thres, iou_thres, 0, False, max_det=max_det)
                # Process predictions
                for i, det in enumerate(pred):  # per image
                    p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
                    p = Path(p)  # to Path
                    save_path = str(result_folder / p.name)  # im.jpg
                    if len(det):
                        # Rescale boxes from img_size to im0 size
                        det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
            read_image = cv2.imread(image_path)
            print(det)





            progress_bar.update(1)
            break


def parse_opt():
    parser = argparse.ArgumentParser(description='Detect name of hero version 1.0.0.0')
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5/runs/train/yolov5s_results8/weights/best.pt', help='model path or triton URL')
    parser.add_argument('--source', type=str, default=ROOT / 'custom_test/test/images', help='Path to folder contains images')
    parser.add_argument('--reference-heroes', type=str, default=ROOT / 'crawled_images', help='Path to folder contains reference heroes')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[416], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--project', default=ROOT / 'runs/detect_hero', help='save results to project/name')
    parser.add_argument('--name', type=str, default='result_', help='Prefix of result folder')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()
    return opt

def main(opt):
    run(**vars(opt))

if __name__ == '__main__':
    opt = parse_opt()
    main(opt)