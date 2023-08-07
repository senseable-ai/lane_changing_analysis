# limit the number of cpus used by high performance libraries
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
sys.path.insert(0, './yolov5')

import argparse
import os
import platform
import shutil
import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn

from yolov5.models.experimental import attempt_load
from yolov5.utils.downloads import attempt_download
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.datasets import LoadImages, LoadStreams
from yolov5.utils.general import (LOGGER, check_img_size, non_max_suppression, scale_coords, 
                                  check_imshow, xyxy2xywh, increment_path)
from yolov5.utils.torch_utils import select_device, time_sync
from yolov5.utils.plots import Annotator, colors
from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort

# Lane finding 2021/12/28 #
###########################
import pyshine as ps
import lane_finding
import numpy as np
import matplotlib.pyplot as plt
###########################

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # yolov5 deepsort root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative



def detect_yellow_coordinates(image, frame_number, fps):
    if image.shape[0] == 0:
        print("Warning: Empty image encountered.")
        return [], None

    # 현재 프레임의 시간을 계산
    current_time_seconds = frame_number / fps
    minutes, seconds = divmod(current_time_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    detection_time = f"{int(hours)}:{int(minutes)}:{int(seconds)}"

    # Convert to the HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define a color range for yellow
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])

    # Create a binary mask where white pixels represent the presence of yellow
    mask = cv2.inRange(hsv_image, lower_yellow, upper_yellow)

    # Find the coordinates of yellow pixels
    yellow_coordinates = np.column_stack(np.where(mask > 0))

    return yellow_coordinates, detection_time
def detect(opt):
    out, source, yolo_model, deep_sort_model, show_vid, save_vid, save_txt, imgsz, evaluate, half, project, name, exist_ok= \
        opt.output, opt.source, opt.yolo_model, opt.deep_sort_model, opt.show_vid, opt.save_vid, \
        opt.save_txt, opt.imgsz, opt.evaluate, opt.half, opt.project, opt.name, opt.exist_ok
    webcam = source == '0' or source.startswith(
        'rtsp') or source.startswith('http') or source.endswith('.txt')

    # initialize deepsort
    cfg = get_config()
    cfg.merge_from_file(opt.config_deepsort)
    deepsort = DeepSort(deep_sort_model,
                        max_dist=cfg.DEEPSORT.MAX_DIST,
                        max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                        use_cuda=True)

    # Initialize
    device = select_device(opt.device)
    half &= device.type != 'cpu'  # half precision only supported on CUDA

    # The MOT16 evaluation runs multiple inference streams in parallel, each one writing to
    # its own .txt file. Hence, in that case, the output folder is not restored
    if not evaluate:
        if os.path.exists(out):
            pass
            shutil.rmtree(out)  # delete output folder
        os.makedirs(out)  # make new output folder


    #####################################
    # Lane Finding Parameter 2021/12/28 #
    #############################################
    init=True
    mtx, dist = lane_finding.distortion_factors()
    #############################################

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    save_dir.mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(yolo_model, device=device, dnn=opt.dnn)
    stride, names, pt, jit, _ = model.stride, model.names, model.pt, model.jit, model.onnx
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Half
    half &= pt and device.type != 'cpu'  # half precision only supported by PyTorch on CUDA
    if pt:
        model.model.half() if half else model.model.float()

    # Set Dataloader
    vid_path, vid_writer = None, None
    # Check if environment supports image displays
    if show_vid:
        show_vid = check_imshow()

    # Dataloader
    if webcam:
        show_vid = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt and not jit)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt and not jit)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names

    # extract what is in between the last '/' and last '.'
    txt_file_name = source.split('/')[-1].split('.')[0]
    txt_path = str(Path(save_dir)) + '/' + txt_file_name + '.txt'


    if pt and device.type != 'cpu':
        model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.model.parameters())))  # warmup
    dt, seen = [0.0, 0.0, 0.0, 0.0], 0
    for frame_idx, (path, img, im0s, vid_cap, s) in enumerate(dataset):
        t1 = time_sync()
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if opt.visualize else False
        pred = model(img, augment=opt.augment, visualize=visualize)
        t3 = time_sync()
        dt[1] += t3 - t2

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, opt.classes, opt.agnostic_nms, max_det=opt.max_det)
        dt[2] += time_sync() - t3

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, _ = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, _ = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg, vid.mp4, ...
            s += '%gx%g ' % img.shape[2:]  # print string

            ##############################################################################
            if seen==1:
                if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                if vid_cap:  # video
                    fps = vid_cap.get(cv2.CAP_PROP_FPS)
                    w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                   
                else:  # stream
                    fps, w, h = 30, im0.shape[1], im0.shape[0]

                ratio = w/200000
                
                vid_writer = cv2.VideoWriter('./yolov5/data/output/test_sample_result.avi', 
                            cv2.VideoWriter_fourcc(*'XVID'), fps, (w, h))
            ################################################################################

            ###########################
            # Lane finding 2021/12/28 #
            ###################################################################################################
            img_out, angle, colorwarp, draw_poly_img = lane_finding.lane_finding_pipeline(im0, init, mtx, dist)

            if angle>1.5 or angle <-1.5:
                init=True
            else:
                init=False
            ###################################################################################################

            annotator = Annotator(img_out, line_width=2, pil=not ascii)

            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(
                    img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                xywhs = xyxy2xywh(det[:, 0:4])
                confs = det[:, 4]
                clss = det[:, 5]

                #initialize color map 2021.12.28
                cmap = plt.get_cmap('tab20b')
                colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

                # pass detections to deepsort
                t4 = time_sync()
                outputs = deepsort.update(xywhs.cpu(), confs.cpu(), clss.cpu(), im0)
                t5 = time_sync()
                dt[3] += t5 - t4

                # draw boxes for visualization
                if len(outputs) > 0:
                    for j, (output, conf) in enumerate(zip(outputs, confs)):

                        bboxes = output[0:4]
                        id = output[4]
                        cls = output[5]

                        # color with cmap  2021.12.28
                        color_ = colors[int(id) % len(colors)]
                        color_ = [i * 255 for i in color_]

                        c = int(cls)  # integer class
                        label = f'{id} {names[c]} {conf:.2f}'
                        
                        # Set default detection color to green (0, 255, 0)
                        detection_color = (0, 255, 0)

                        # Calculate the bounding box center
                        mid_x = (bboxes[0] + bboxes[2]) / 2
                        mid_y = (bboxes[1] + bboxes[3]) / 2

                        # Assuming the bottom of the image as the position of my car
                        my_car_y = h  # h is the height of the image

                        # Estimate the distance using the y-coordinate difference
                        distance = round(((h - bboxes[3]) * ratio) * 2, 1)  # Using the existing distance formula

                        # If the distance is less than or equal to 5, change the color to red (0, 0, 255)
                        if distance < 8:
    
                            # If the distance is less than or equal to 5, change the color to red (0, 0, 255)
                            if distance <= 5:
                                detection_color = (0, 0, 255)

                        annotator.box_label(bboxes, label, color=detection_color)

                        
                        ###################################
                        # Distance Calculation 2021/12/28 #
                        ########################################################
                        #    car :3         bus        truck
                        if cls == 2 :
                            
                            mid_x = (bboxes[0]+bboxes[2])/2
                            mid_y = (bboxes[1]+bboxes[3])/2
                            apx_distance = round((((h-bboxes[3]))*ratio)*2,1)
                            mid_xy =[mid_x,mid_y]
                            # update for text input
                        if distance < 8:
                                annotator.dist(mid_xy,apx_distance)

                        if apx_distance <= 1:
                            if (mid_x) > w*0.3 and (mid_x) < w*0.7:
                                warn_xy =[400,150]
                                annotator.dist(warn_xy,cls,id)
                        #########################################################
                        
                        if save_txt:
                            # to MOT format
                            bbox_left = output[0]
                            bbox_top = output[1]
                            bbox_w = output[2] - output[0]
                            bbox_h = output[3] - output[1]
                        # Crop the detected bounding box using bbox_left, bbox_top, bbox_w, bbox_h
                        crop_image = im0[int(bbox_top):int(bbox_top + bbox_h), int(bbox_left):int(bbox_left + bbox_w)]
                        frame_number = int(vid_cap.get(cv2.CAP_PROP_POS_FRAMES)) # 현재 프레임 번호
                        fps = vid_cap.get(cv2.CAP_PROP_FPS) # 동영상의 프레임 속도
                        yellow_coordinates, detection_time = detect_yellow_coordinates(crop_image, frame_number, fps)

                        # Calculate the center coordinates of the bounding box
                        center_x = bbox_left + bbox_w / 2
                        center_y = bbox_top + bbox_h / 2

                        # Calculate the distance between the bounding box center and the yellow coordinates (if detected)
                        
                        # if len(yellow_coordinates) > 0:
                        #     yellow_x, yellow_y = yellow_coordinates[0]  # Assuming the first detected yellow coordinate
                        #     distance = ((center_x - yellow_x) ** 2 + (center_y - yellow_y) ** 2) ** 0.5

                        # Save class ID, distance, status, and detection time to text file
                        txt_dir = save_dir / 'crops' / names[c]
                        txt_dir.mkdir(parents=True, exist_ok=True)  # Create the directory if it doesn't exist
                        txt_path = txt_dir / f'{p.stem}_yellow_coordinates.txt'
                        with open(txt_path, 'a') as f:
                            
                            if 0<= distance <= 5: #차간 거리가 5m 이하일 때
                                if len(yellow_coordinates) > 0: #깜빡이가 검출되면
                                    f.write(f'{frame_number} {int(id)} {bbox_left} {bbox_top} {bbox_w} {bbox_h} 1 {distance} 1 {detection_time}\n') #깜빡이 검출 1 거리 5 이하 1

                            if 0 <= distance <= 5: #차간 거리가 5m 이하일 때
                                if len(yellow_coordinates) == 0: #깜빡이가 검출되지 않으면
                                    f.write(f'{frame_number} {int(id)} {bbox_left} {bbox_top} {bbox_w} {bbox_h} 0 {distance} 1 {detection_time}\n') #깜빡이 검출 9 거리 5 이하 1

                            if distance > 5: #차간 거리가 5m 초과 일때
                                if len(yellow_coordinates) > 0: #깜빡이가 검출되면 
                                    f.write(f'{frame_number} {int(id)} {bbox_left} {bbox_top} {bbox_w} {bbox_h} 1 {distance} 0 {detection_time}\n') #깜빡이 검출 0 거리 5초과 0
                            
                            if distance > 5: #차간 거리가 5m 초과 일때 
                                if len(yellow_coordinates) == 0: # 깜빡이가 검출되지 않으면
                                    f.write(f'{frame_number} {int(id)} {bbox_left} {bbox_top} {bbox_w} {bbox_h} 0 {distance} 0 {detection_time}\n') #깜빡이 검출 0 거리 5초과 0

                        

                        # # Save yellow coordinates and detection time to text file
                        # if len(yellow_coordinates) > 0:
                        #     txt_dir = save_dir / 'crops' / names[c]
                        #     txt_dir.mkdir(parents=True, exist_ok=True)  # Create the directory if it doesn't exist
                        #     txt_path = txt_dir / f'{p.stem}_yellow_coordinates.txt'
                        #     with open(txt_path, 'a') as f:
                        #         for coordinate in yellow_coordinates:
                        #             f.write(f'{frame_number} {int(cls)} {bbox_left} {bbox_top} {bbox_w} {bbox_h} 1 {detection_time}\n')

                          

                LOGGER.info(f'{s}Done. YOLO:({t3 - t2:.3f}s), DeepSort:({t5 - t4:.3f}s)')

            else:
                deepsort.increment_ages()

            # Print time (inference-only)
            #LOGGER.info(f'{s}Done. YOLO:({t3 - t2:.3f}s), DeepSort:({t5 - t4:.3f}s)')
                LOGGER.info('No detections')

            # Stream results
            im0 = annotator.result()
            if show_vid:
                #cv2.imshow(p, im0)
                cv2.imshow("result", im0)
                cv2.imshow(str(p), im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

            # Save results (image with detections)
            if save_vid:
                if vid_path != save_path:  # new video
                    vid_path = save_path
                    if isinstance(vid_writer, cv2.VideoWriter):
                        vid_writer.release()  # release previous video writer
                    if vid_cap:  # video
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    else:  # stream
                        fps, w, h = 30, im0.shape[1], im0.shape[0]

                    vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

                vid_writer.write(im0)

            vid_writer.write(im0)
            ########################

    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    #LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS, %.1fms deep sort update \
    #    per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_vid:
        print('Results saved to %s' % save_path)
        if platform == 'darwin':  # MacOS
            os.system('open ' + save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo_model', nargs='+', type=str, default='yolov5m.pt', help='model.pt path(s)')
    parser.add_argument('--deep_sort_model', type=str, default='osnet_x0_25')
    parser.add_argument('--source', type=str, default='C:\\Users\\user\\Desktop\\im_jg_DeepSort_Pytorch_lane_detection\\yolov5\\data\\video\\NOR_20230612_093659_FT.mp4', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.3, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--show-vid', action='store_true', help='display tracking video results')
    parser.add_argument('--save-vid', action='store_true', help='save video tracking results')
    parser.add_argument('--save-txt', action='store_true', help='save MOT compliant results to *.txt')
    # class 0 is person, 1 is bycicle, 2 is car... 79 is oven
    parser.add_argument('--classes', nargs='+', type=int, default=[2], help='filter by class: --class 0, or --class 16 17')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--evaluate', action='store_true', help='augmented inference')
    parser.add_argument("--config_deepsort", type=str, default="deep_sort/configs/deep_sort.yaml")
    parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detection per image')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--project', default=ROOT / 'runs/track', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand

    with torch.no_grad():
        detect(opt)