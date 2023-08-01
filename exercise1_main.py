import argparse
import os
import platform
import sys
from pathlib import Path
import glob
import re
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
from sklearn.metrics import classification_report

from my_utils import helper, image_processing

def run(
        weights=ROOT / 'yolov5/runs/train/yolov5s_results8/weights/best.pt',
        source=ROOT / 'yolov5/custom_test/valid/images',
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
        save_img=True
):
    source = str(source)
    result_folder = helper.increment_path(Path(project) / name, exist_ok=exist_ok)
    report_name = "output"
    report_file = helper.generate_treefolder(report_name, str(result_folder))

    # YoloV5
    device = select_device(device)
    image_paths_list = []
    if os.path.isfile(source):
        image_paths_list.append(source)
    elif os.path.isdir(source):
        image_paths_list = glob.glob(source+"/*.png", recursive=True) + glob.glob(source+"/*.jpg", recursive=True) 

    wild_rift_heroes = glob.glob(str(reference_heroes)+"/*.png", recursive=True) + glob.glob(str(reference_heroes)+"/*.jpg", recursive=True)
    print(image_paths_list)

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    model.warmup(imgsz=(1, 3, *imgsz))

    output_result = []
    y_test, y_predict = [], []
    true_positives = 0

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
                    annotator = Annotator(im0, line_width=line_thickness, example=str(names))
                    if len(det):
                        # Rescale boxes from img_size to im0 size
                        det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
                        
                        # Write results
                        for *xyxy, conf, cls in reversed(det):
                            # Add bbox to image
                            c = int(cls)  # integer class
                            label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                            annotator.box_label(xyxy, label, color=colors(c, True))
                    # Stream results
                    im0 = annotator.result()
                    if save_img:
                        cv2.imwrite(save_path, im0)
                        print(f"Save detected to: {save_path}")

            # Skip image if there are no detected boxes
            if len(det)==0: 
                output_result.append('\t'.join([image_path.split('\\')[-1], 'Hero_not_found']))
                continue

            # Box format: (x1,y1,x2,y2,confidence)
            detected_boxes = sorted(det, key=lambda x:x[0])
            mostleft_detected_box = detected_boxes[0]
            read_image = cv2.imread(image_path)

            # Skip image if there isnt any heroes in the left side of the image
            if mostleft_detected_box[0] > read_image.shape[1]/2:
                output_result.append('\t'.join([image_path.split('\\')[-1], 'Hero_not_found']))
                continue

            # Mapping between detected hero with reference heroes
            similarity_scores_list = []
            actual_image = read_image[int(mostleft_detected_box[1]): int(mostleft_detected_box[3]),
                                    int(mostleft_detected_box[0]):int(mostleft_detected_box[2])]
            actual_image = cv2.resize(actual_image, (50, 50))

            for ref_hero in wild_rift_heroes:
                ref_image = cv2.imread(ref_hero)
                ref_image = cv2.resize(ref_image, (50, 50))
                score = image_processing.calculate_structural_similarity(actual_image, ref_image)
                similarity_scores_list.append([ref_hero, score])
            
            # Ouput the highest score hero
            similarity_scores_list = sorted(similarity_scores_list, key=lambda x:x[1], reverse=True)
            predicted_hero_name = similarity_scores_list[0][0].split('\\')[-1]
            predicted_hero_name = predicted_hero_name.split('\\')[-1].split('_OriginalSquare')[0]
            predicted_hero_name = re.sub(r'[^a-zA-Z._ ]', '',string=predicted_hero_name)
            actual_hero_name = image_path.split('\\')[-1]
            
            if predicted_hero_name in actual_hero_name:
                true_positives += 1

            output_result.append('\t'.join(
                [actual_hero_name,
                 predicted_hero_name]
            ))

            # Append to y_test and y_predict
            actual_hero_name = re.search(r'^([^\d]+)_', string=actual_hero_name).group().rstrip('_')
            actual_hero_name = re.sub('Jarvan', 'Jarvan_IV', actual_hero_name)
            actual_hero_name = re.sub('Lee', 'Lee_Sin', actual_hero_name)
            y_test.append(actual_hero_name)
            y_predict.append(predicted_hero_name)

            progress_bar.update(1)
    if len(y_test) == len(y_predict) !=0:
        report = classification_report(y_test, y_predict)
    else:
        report = 'Data not found'
    with open(report_file, 'w', encoding='utf-8') as f:
        f.writelines('\n'.join(output_result))
        f.write('\n******** Hero name recognitions Report ********')
        # f.write(f'\nAccuracy: {true_positives/len(image_paths_list)}')
        f.write(f'\nConfusion matrix: \n {report}')
    print(f"Saved to: {report_file}")
    print(report)


def parse_opt():
    parser = argparse.ArgumentParser(description='Detect name of hero version 1.0.0.0')
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5/runs/train/yolov5s_results8/weights/best.pt', help='model path or triton URL')
    parser.add_argument('--source', type=str, default=ROOT / 'custom_test/test_custom/images', help='Path to folder contains images')
    parser.add_argument('--reference-heroes', type=str, default=ROOT / 'crawled_images', help='Path to folder contains reference heroes')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=(416, 416), help='inference size h,w')
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