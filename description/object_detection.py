import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ultralytics import YOLO

class ObjectDetectionNode(Node):
    def __init__(self):
        super().__init__('object_detection_node')
        
        # Initialize CvBridge to convert ROS messages to OpenCV images
        self.bridge = CvBridge()
        
        # Create a subscriber for the camera feed
        self.subscription = self.create_subscription(
            Image,                       # Message type
            '/camera/image_raw',         # Topic name (your camera topic)
            self.image_callback,         # Callback function
            10                           # QoS (Quality of Service) setting
        )
        
        # Create a publisher for the processed image
        self.publisher = self.create_publisher(
            Image,                       # Message type
            '/camera/image_with_detections', # Topic name for processed image
            10                           # QoS setting
        )
        
        # Initialize YOLO model
        self.model = YOLO('yolov5s.pt')

    def image_callback(self, msg):
        try:
            # Convert the ROS image message to OpenCV format
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            # Run object detection
            results = self.model(frame)
            
            # Draw bounding boxes and labels on the frame
            for result in results:
                boxes = result.boxes.xyxy.numpy()  # Bounding box coordinates
                confidences = result.boxes.conf.numpy()  # Confidence scores
                labels = result.boxes.cls.numpy()  # Class indices
                
                for box, conf, label in zip(boxes, confidences, labels):
                    # Convert coordinates to integers
                    x1, y1, x2, y2 = map(int, box)
                    
                    # Get the label name from YOLO's class list
                    label_name = self.model.names[int(label)]
                    
                    # Draw the bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Add the label and confidence
                    cv2.putText(frame, f"{label_name} {conf:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Convert the processed frame back to a ROS Image message
            output_msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
            
            # Publish the processed image
            self.publisher.publish(output_msg)

        except Exception as e:
            self.get_logger().info(f"Error processing image: {str(e)}")

def main(args=None):
    rclpy.init(args=args)
    node = ObjectDetectionNode()
    
    # Spin the node to keep it running
    rclpy.spin(node)
    
    # Clean up
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
