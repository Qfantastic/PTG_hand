import sys
import threading
import cv2
import os

import rospy
import message_filters
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

# import torch


class ImageListener:
    def __init__(
        self,
        topic_list,
        node_id="image_listener",
        synced_slop=0.1,
    ):
        self._cv_bridge = CvBridge()
        self._lock = threading.Lock()
        self._node_id = node_id
        self._topic_list = topic_list
        self._synced_queue_size = len(topic_list)
        self._synced_slop = synced_slop
        self._synced_msgs = None
        self.prev_ts = 0
        self.count = 0
        self.frames_bundle = []
        self._fourcc = cv2.VideoWriter_fourcc(*'mp4v') 

        # initialize a node
        rospy.init_node("ImageListener", anonymous=True)

        self._fs = [
            message_filters.Subscriber(t, Image, queue_size=10)
            for t in self._topic_list
        ]

        ts = message_filters.ApproximateTimeSynchronizer(
            fs=self._fs,
            queue_size=self._synced_queue_size,
            slop=self._synced_slop,
        )
        ts.registerCallback(self.ts_callback)

    def ts_callback(self, *msgs):
        self._synced_msgs = msgs

    def run(self):
        # pass
        video_count=0
        while not rospy.is_shutdown():

            if self._synced_msgs is not None:
                color_msg = self._synced_msgs[0]
                curr_ts = color_msg.header.stamp.to_nsec()
                if curr_ts != self.prev_ts:
                    self.prev_ts = curr_ts
                    self.count += 1
                    frame_rgb = self._cv_bridge.imgmsg_to_cv2(color_msg)[:,:,::-1].copy()
                    self.frames_bundle.append(
                        # (
                        #     torch.from_numpy(frame_rgb),
                        #     curr_ts,
                        # )
                        self._cv_bridge.imgmsg_to_cv2(color_msg)
                    )
                    # print(f"count: {self.count} \t {curr_ts}")
                    if self.count == 90: # 90 frames
                        print("90 frames received")
                        self.count = 0
                        ####################
                        # run the network function for 
                        # self.frames_bundle[]
                        self.generate_mp4_video(
                            file_name = os.path.join(
                                os.path.dirname(__file__),
                                f"{video_count}.mp4"
                            )
                        )
                        video_count+=1
                        ####################
                        self.frames_bundle=[]
                    cv2.imshow(self._node_id, self._cv_bridge.imgmsg_to_cv2(color_msg))
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

        cv2.destroyAllWindows()

    def generate_mp4_video(self, file_name):
        height, width, _ = self.frames_bundle[0].shape
        video = cv2.VideoWriter(file_name, self._fourcc, 30, (width, height))
        for img in self.frames_bundle:
            video.write(img)
        video.release()


if __name__ == "__main__":

    # image listener
    listener = ImageListener(
        topic_list=[
            "/hololens2/sensor_color/image_raw",
            # "/hololens2/sensor_depth/image_raw",
        ]
    )
    listener.run()
