import rclpy
from rclpy.node import Node
import sounddevice as sd
import numpy as np
from vosk import Model, KaldiRecognizer
import json
from geometry_msgs.msg import Twist

class VoiceControlledRover(Node):
    def __init__(self):
        super().__init__('voice_controlled_rover')

        # Initialize the Vosk model
        self.model = Model("/home/chandu/Downloads/vosk-model-small-en-us-0.15")
        
        # Initialize the Kaldi Recognizer for speech recognition (16kHz audio)
        self.recognizer = KaldiRecognizer(self.model, 16000)

        # Set up the audio stream from the microphone
        self.stream = sd.RawInputStream(
            samplerate=16000, blocksize=8000, dtype='int16', channels=1, callback=self.audio_callback
        )
        self.stream.start()

        # Publisher for the /cmd_vel topic to control rover movement
        self.cmd_vel_publisher = self.create_publisher(Twist, '/cmd_vel', 10)

        self.get_logger().info("Voice-controlled rover node initialized.")

    def audio_callback(self, indata, frames, time, status):
        """This method will be called when audio data is received."""
        if status:
            self.get_logger().warn(f"Audio stream status: {status}")
        
        # Convert the cffi buffer to a byte object
        audio_data = memoryview(indata).tobytes()

        # Accept the incoming audio data and perform speech recognition
        if self.recognizer.AcceptWaveform(audio_data):
            result = self.recognizer.Result()  # Returns a JSON string with recognized speech
            print(result)

            # Parse the result to get the recognized text
            result_json = json.loads(result)
            if 'text' in result_json:
                command = result_json['text']
                self.get_logger().info(f"Recognized Command: {command}")
                self.process_command(command)

    def process_command(self, command):
        """Process the recognized voice command and control the rover."""
        command = command.lower()

        # Create a new Twist message to send velocity commands
        twist = Twist()

        if "forward" in command:
            self.get_logger().info("Moving forward")
            twist.linear.x = 0.5  # Move forward with 0.5 m/s
            twist.angular.z = 0.0  # No turning
        elif "backward" in command:
            self.get_logger().info("Moving backward")
            twist.linear.x = -0.5  # Move backward with -0.5 m/s
            twist.angular.z = 0.0  # No turning
        elif "left" in command:
            self.get_logger().info("Turning left")
            twist.linear.x = 0.0  # No forward/backward movement
            twist.angular.z = 0.5  # Turn left with 0.5 rad/s
        elif "right" in command:
            self.get_logger().info("Turning right")
            twist.linear.x = 0.0  # No forward/backward movement
            twist.angular.z = -0.5  # Turn right with -0.5 rad/s
        elif "stop" in command:
            self.get_logger().info("Stopping rover")
            twist.linear.x = 0.0  # Stop moving
            twist.angular.z = 0.0  # Stop turning
        else:
            self.get_logger().info(f"Unrecognized command: {command}")

        # Publish the velocity command to /cmd_vel
        self.cmd_vel_publisher.publish(twist)

def main(args=None):
    rclpy.init(args=args)
    node = VoiceControlledRover()

    # Spin the node to keep it running
    rclpy.spin(node)

    # Clean up
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
