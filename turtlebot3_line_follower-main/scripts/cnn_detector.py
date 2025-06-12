# scripts/cnn_detector.py
#!/usr/bin/env python3
import rospy
import cv2
import torch
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from models.cnn_model import LineFollowerCNN

class CNNDetector:
    def __init__(self):
        self.bridge = CvBridge()
        self.model = LineFollowerCNN()
        self.model.load_state_dict(torch.load('models/model_weights.pth'))
        self.model.eval()
        
        # ROS Setup
        self.image_sub = rospy.Subscriber("/camera/image_raw", Image, self.img_callback)
        self.error_pub = rospy.Publisher("/cnn_error", Float32, queue_size=1)

    def img_callback(self, msg):
        cv_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        processed = self.preprocess(cv_img)  # Resize + normalize
        with torch.no_grad():
            error = self.model(processed)
        self.error_pub.publish(error.item())

    def preprocess(self, img):
        img = cv2.resize(img, (320, 240))
        img = torch.from_numpy(img).float().permute(2,0,1)/255.0
        return img.unsqueeze(0)  # Add batch dimension

if __name__ == "__main__":
    rospy.init_node('cnn_detector')
    detector = CNNDetector()
    rospy.spin()