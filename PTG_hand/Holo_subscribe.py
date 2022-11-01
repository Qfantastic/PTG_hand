# from tomlkit import key
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.optim as optim
from torchvision import datasets, models, transforms

import message_filters
import cv2
# import threading
import argparse
import numpy as np
import rospy

from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import PIL
# lock = threading.Lock()



# import argparse

import sys
from pathlib import Path


# import colorsys




import os
# os.environ['PYOPENGL_PLATFORM'] = 'egl'# 'osmesa'

from torchvision.utils import make_grid
import trimesh
# import math

from utils.augmentations import letterbox
from utils.torch_utils import select_device, time_sync
from models.common import DetectMultiBackend
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box

class_names = np.array(['coffee_bean', 'dropper', 'filter', 'grinder', 'grinder_lip', 'kettle',
                        'kettle_lip', 'measure_cup', 'mug', 'only_hand', 'thermometer', 'weighter'])
tf_avg = np.load('./data/tf_avg.npy')
depth_h = 512
depth_w = 512
curr_dir = os.path.dirname(os.path.abspath(__file__))
lut_file = os.path.join(curr_dir, "./data/depth_lut.bin")
with open(lut_file, mode="rb") as lut_file:
    lut = np.frombuffer(lut_file.read(), dtype="f")
    lut = np.reshape(lut, (-1, 3))

def world2pixel_box(x, intrinsic):
    fx = intrinsic[0]
    fy = intrinsic[1]
    ux = intrinsic[2]
    uy = intrinsic[3]

    x[:, 0] = x[:, 0] * fx / x[:, 2] + ux
    x[:, 1] = x[:, 1] * fy / x[:, 2] + uy
    return x[:, 0:2]

def world2pixel(x,intrinsic):
    t = x.copy()
    fx = intrinsic[0]
    fy = intrinsic[1]
    ux = intrinsic[2]
    uy = intrinsic[3]

    t[0] = x[0] * fx / x[2] + ux
    t[1] = x[1] * fy / x[2] + uy
    return t[0:2]

def bounding_box_to_pixel(x, intrinsic, range_box):
    sign = [1, -1]
    bounding_box_w = np.zeros((8, 3))
    bounding_box_p = np.zeros((4, 2))
    n = 0
    for i in range(2):
        for j in range(2):
            for m in range(2):
                bounding_box_w[n, 0] = x[0] + sign[i] * range_box
                bounding_box_w[n, 1] = x[1] + sign[j] * range_box
                bounding_box_w[n, 2] = x[2] + sign[m] * 0.6 * range_box
                n = n + 1

    result = world2pixel_box(bounding_box_w, intrinsic)

    l_x = np.zeros(8)
    l_y = np.zeros(8)
    for j in range(8):
        l_x[j] = result[j, 0]
        l_y[j] = result[j, 1]
    bound_x_min = np.min(l_x)
    bound_x_max = np.max(l_x)
    bound_y_min = np.min(l_y)
    bound_y_max = np.max(l_y)

    bounding_box_p[0, :] = np.array([bound_x_min, bound_y_min])
    bounding_box_p[1, :] = np.array([bound_x_max, bound_y_min])
    bounding_box_p[2, :] = np.array([bound_x_max, bound_y_max])
    bounding_box_p[3, :] = np.array([bound_x_min, bound_y_max])
    return bounding_box_p


def holo_2D_to_3D(abc, lut, center, depth_image_or, intrinsic, range_box, tf, abc_or,model_cls, device, pre_process):
    height_color = int(abc.shape[0])
    width_color = int(abc.shape[1])

    depth_image = depth_image_or / 1000
    depth_image = np.tile(depth_image.flatten().reshape((-1, 1)), (1, 3))
    points = depth_image * lut
    n_center = center[1] * depth_w + center[0]
    hand_center_3D_depth = np.append(points[n_center],1)
    hand_center_3D_color = np.dot(tf, hand_center_3D_depth)[0:3]

    bounding_box_p = bounding_box_to_pixel(hand_center_3D_color, intrinsic, range_box)
    center_p = world2pixel(hand_center_3D_color, intrinsic)

    image_to_draw = abc.copy()
    # cv2.rectangle(image_to_draw, (int(bounding_box_p[0,0]),int(bounding_box_p[0,1])),
    #               (int(bounding_box_p[2,0]),int(bounding_box_p[2,1])),(0,255,0),1)

    c =0.75

    x1 = int(bounding_box_p[0][0])
    y1 = int(bounding_box_p[0][1])
    x2 = int(bounding_box_p[2][0])
    y2 = int(bounding_box_p[2][1])

    w = x2 - x1
    h = y2 - y1

    if ((width_color-x1)>c*w and (height_color-y1)>c*h and x2>c*w and y2>c*h):
        annotator = Annotator(abc_or.copy(), line_width=3)
        crop_color = abc_or[max(y1,0):min(y2,height_color),max(x1,0):min(x2,width_color)]


        crop_color = PIL.Image.fromarray(crop_color)
        crop_color = pre_process(crop_color)
        crop_color = crop_color.unsqueeze(0)
        crop_color = crop_color.to(device)
        outputs = model_cls(crop_color)
        _, pre_crop = torch.max(outputs.data, 1)
        class_index = int(pre_crop.cpu().numpy())
        print(class_names[class_index])
        cv2.rectangle(image_to_draw, (max(x1,0),max(y1,0)),
                      (min(x2,width_color),min(y2,height_color)),(0,0,255),3)
        cv2.putText(image_to_draw, class_names[class_index], (max(x1,0),max(y1,0)), cv2.FONT_HERSHEY_SIMPLEX,
                    1,(255,0,0),3,cv2.LINE_AA)


    print('center_p', center_p)
    return image_to_draw
    # plt.figure()
    # plt.imshow(abc)
    # plt.plot(center_p[0], center_p[1], 'r+')
    #
    # for i in range(3):
    #     plt.plot([bounding_box_p[i][0], bounding_box_p[i + 1][0]],
    #              [bounding_box_p[i][1], bounding_box_p[i + 1][1]], color='green')
    #
    # plt.plot([bounding_box_p[0][0], bounding_box_p[3][0]],
    #          [bounding_box_p[0][1], bounding_box_p[3][1]], color='green')
    #
    # save_path = './result_0827_' + args.folder + '/test_holo_color/'
    #
    # if not os.path.isdir(save_path):
    #     os.makedirs(save_path)
    # plt.savefig(os.path.join(save_path, 'result') + str(n_image) + '.png', dpi=200)
    # print('saving image %i' % n_image)
    # plt.close()

class ImageListener:

    def __init__(self, topics, node_id="image_listener",slop_seconds=0.1):

        self.cv_bridge = CvBridge()
        self.node_id = node_id
        self.topics = topics
        
        #self.queue_size = 2 * len(topics)
        self.queue_size = 3
        self.slop_seconds = slop_seconds
        # self.im = None
        # self.depth_or = None
        # self.depth = None
        # self.depth_frame_id = None
        # self.depth_frame_stamp = None
        self.empty_label = np.zeros((176, 176, 3), dtype=np.uint8)
        
        self.synced_msgs = None


        # initialize a node
        rospy.init_node(self.node_id, anonymous=True)

        self.box_pub = rospy.Publisher('box_label', Image, queue_size=10)
        self.color_box_pub = rospy.Publisher('color_box_label', Image, queue_size=10)
        self.holo_subs = [
            message_filters.Subscriber(t, Image, queue_size=10)
            for t in topics[:-1]       
        ]
        self.holo_subs.append(
            message_filters.Subscriber(topics[-1], CameraInfo, queue_size=10)
        )

        ts = message_filters.ApproximateTimeSynchronizer(
            self.holo_subs, 
            self.queue_size, 
            self.slop_seconds
        )
        ts.registerCallback(self.ts_callback)

    def ts_callback(self, *msg):
        self.synced_msgs = msg
        # if depth.encoding == '32FC1':
        #     print('32')
        #     depth_cv = self.cv_bridge.imgmsg_to_cv2(depth)
        # elif depth.encoding == '16UC1':
        #     depth_cv = self.cv_bridge.imgmsg_to_cv2(depth)
        #     #depth_cv = self.cv_bridge.imgmsg_to_cv2(depth).copy().astype(np.float32)
        #     #depth_cv = depth_cv/1000.0
        # else:
        #     print('else')
        #     rospy.logerr_throttle(
        #         1, 'Unsupported depth type. Expected 16UC1 or 32FC1, got {}'.format(
        #             depth.encoding))
        #     return

      

        # with lock:
     
        #     self.depth_or = depth_cv.copy()
        #     #self.depth = depth_cv.copy()
        #     self.depth_frame_id = depth.header.frame_id
        #     self.depth_frame_stamp = depth.header.stamp


    

    def run_network(self, model, index, model_cls, device , pre_process,imgsz=(640, 640), augment=False, visualize=False, conf_thres=0.45,
                    iou_thres=0.4, classes=None, agnostic_nms=False, max_det=1000,line_thickness=3,hide_labels=False,
                    hide_conf=False):

        # with lock:
        #     # if listener.im is None:
        #     #     return
        #     if self.depth_or is None:
        #         return
   
        #     depth_or_img = np.array(self.depth_or).copy()
        #     #depth_img = np.array(self.depth_or).copy()
        #     depth_frame_id = self.depth_frame_id
        #     depth_frame_stamp = self.depth_frame_stamp
        
        stride, names, pt = model.stride, model.names, model.pt
        imgsz = check_img_size(imgsz, s=stride)

        if self.synced_msgs is not None:
            depth_msg, color_msg, caminfo_msg = self.synced_msgs
            depth_or_img = self.cv_bridge.imgmsg_to_cv2(depth_msg,depth_msg.encoding).astype('uint16')
            depth_frame_id = depth_msg.header.frame_id
            depth_frame_stamp = depth_msg.header.stamp

            color_or_img = self.cv_bridge.imgmsg_to_cv2(color_msg,color_msg.encoding)
            color_frame_id = color_msg.header.frame_id
            color_frame_stamp = color_msg.header.stamp

            K_mat = np.array(caminfo_msg.P, dtype=np.float32).reshape((3,4))[:3,:3]
            print(K_mat)

        # if(depth_or_img.ndim == 2):
            print('saving images ',index)
            #cv2.imwrite('test/images2/'+str(index)+'.png',depth_or_img)

            # box_msg = self.cv_bridge.cv2_to_imgmsg(depth_or_img)
            # box_msg.header.stamp = depth_frame_stamp
            # box_msg.header.frame_id = depth_frame_id
            # box_msg.encoding = '16UC1'
            # self.box_pub.publish(box_msg)

            print('max:', np.max(depth_or_img), 'min:', np.min(depth_or_img))
            range_img = np.max(depth_or_img) - np.min(depth_or_img)
            depth_img = ((depth_or_img/range_img)*255).astype('uint8')
            # depth_or_img[depth_or_img > 1000] = 0
            # depth_img = ((depth_or_img/1000.0)*255).astype('uint8')


            #print(depth_img.ndim)

            stacked_img = np.stack((depth_img,) * 3, axis=-1)
            print('max:', np.max(stacked_img), 'min:', np.min(stacked_img))

            #cv2.imwrite('test/images_holo_0/'+str(index)+'.png',stacked_img)

            #print(stacked_img.shape)
            img = letterbox(stacked_img, imgsz, stride=stride, auto=pt)[0]



            img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
            #print('max:', np.max(img), 'min:', np.min(img))
            img = np.ascontiguousarray(img)

            bs = 1
            model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))

            #print('max:', np.max(img), 'min:', np.min(img))
            img = torch.from_numpy(img).to(device)
            img = img.half() if model.fp16 else img.float()  # uint8 to fp16/32
            img /= 255  # 0 - 255 to 0.0 - 1.0



            if len(img.shape) == 3:
                img = img[None]
            print(img.shape)

            pred = model(img, augment=augment, visualize=visualize)
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)


            print('prediction:')


            print(pred)



            for i, det in enumerate(pred):

                im0 = stacked_img.copy()
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
                annotator = Annotator(im0, line_width=line_thickness, example=str(names))
                color_abc = cv2.cvtColor(color_or_img.copy(), cv2.COLOR_BGR2RGB)
                color_or_img_rgb = cv2.cvtColor(color_or_img.copy(), cv2.COLOR_BGR2RGB)
                if len(det):
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()


                    for *xyxy, conf, cls in reversed(det):

                        c = int(cls)
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')

                        center_x = int(xywh[0] * depth_w)
                        center_y = int(xywh[1] * depth_h)
                        center = np.array([center_x, center_y])
                        print('xyxy:',xyxy)
                        annotator.box_label(xyxy, label, color=colors(c,True))
                        holo_color_intrinsic = np.array([K_mat[0][0], K_mat[1][1], K_mat[0][2], K_mat[1][2]])

                        color_abc = holo_2D_to_3D(color_abc, lut, center, depth_or_img, holo_color_intrinsic, 0.08,
                                                  tf_avg,color_or_img_rgb,model_cls, device, pre_process)

                        # color_xyxy , color_label, color_cls = holo_2D_to_3D(color_abc, lut, center, depth_or_img, holo_color_intrinsic, 0.10,
                        #                           tf_avg, color_or_img_rgb, model_cls, device, pre_process)

                im0 = annotator.result()

                #cv2.imwrite('test/images_holo_0_result/'+str(index)+'.png', im0)

                        # xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                        # x_center = xywh[0]*448
                        # y_center = xywh[1]*450
                        # width = xywh[2]*448
                        # height = xywh[3]*450
                        #
                        # x1 = int(x_center-(width/2))
                        # y1 = int(y_center-(height/2))
                        #
                        # x2 = int(x_center+(width/2))
                        # y2 = int(y_center+(height/2))


                        #depth_img8 = (depth_or_img / 255).astype('uint8')
                        # depth_img8UC3= np.stack((depth_img8,) * 3, axis=-1)
                        #image_to_draw = cv2.cvtColor(depth_img8, cv2.COLOR_GRAY2RGB)
                        #image_to_draw = stacked_img.copy()   ************
                        #print(image_to_draw.shape)

                        #cv2.rectangle(image_to_draw, (x1, y1), (x2,y2), (255, 0, 0), 1)


                        #bbox_msg = self.cv_bridge.cv2_to_imgmsg(image_to_draw.astype(np.uint8))
                bbox_msg = self.cv_bridge.cv2_to_imgmsg(im0)
                bbox_msg.header.stamp = depth_frame_stamp
                bbox_msg.header.frame_id = depth_frame_id
                bbox_msg.encoding = 'rgb8'
                self.box_pub.publish(bbox_msg)

                color_bbox_msg = self.cv_bridge.cv2_to_imgmsg(color_abc)
                color_bbox_msg.header.stamp = depth_frame_stamp
                color_bbox_msg.header.frame_id = depth_frame_id
                color_bbox_msg.encoding = 'rgb8'
                self.color_box_pub.publish(color_bbox_msg)

      

                        # cv2.rectangle()





       # print(depth_img[0])



        #abc = depth_img.reshape((450,448,3))
        # print(abc)
        #img = letterbox(depth_img, img_size, stride=stride, auto=pt)[0]








FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

print(ROOT)

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='holo_ROS')
    parser.add_argument('--save_path', dest='save_path', help='Path to save results', default='output/', type=str)
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'best.pt', help='model path(s)')
    parser.add_argument('--data', type=str, default=ROOT / 'data/depth_hand.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    args = parser.parse_args()
    return args



if __name__ == '__main__':



    #network.eval()
    args = parse_args()

    device = select_device(args.device)
    model = DetectMultiBackend(args.weights, device=device, dnn=args.dnn, data=args.data, fp16=args.half)
    print("Loading the Classification Model......")
    model_cls = torch.load("./test/weights_cls/whole_model_1028.pt")
    model_cls.eval()
    pre_process = transforms.Compose([transforms.Resize(224),
                                      transforms.CenterCrop(224),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])



    # image listener
    listener = ImageListener(
        topics = [
            "/hololens2/sensor_depth/image_raw",
            "/hololens2/sensor_color/image_raw",
            "/hololens2/sensor_color/camera_info",
        ]
    )
    index = 0
    while not rospy.is_shutdown():
       listener.run_network(model,index=index ,model_cls = model_cls, device =device , pre_process = pre_process,augment=args.augment)
       index = index + 1
    #listener.write_video('test_box.mp4', 'test_label.mp4')
