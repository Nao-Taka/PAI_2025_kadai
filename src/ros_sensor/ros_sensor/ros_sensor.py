import threading
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, String, Float32MultiArray

#ROSによる通信
#Publisher
class sensor(Node):
    '''
    センサーを発信するためのクラス
    topic_nameを設定してノードを作成する

    runでセンサー動作を開始
    set_valueで値をセット
    killで終了させることも可能
    '''
    def __init__(self, topic_name='/ros_sensor'):
        super().__init__('sensor')
        self.value = ''
        self.pub = self.create_publisher(String, str(topic_name), 10)
        self.timer = self.create_timer(1.0, self.timer_callback)

        self._spin_thread = threading.Thread(target=self.__spin, daemon=True)

    def timer_callback(self):
        text = self.value
        msg = String()
        msg.data = text
        self.pub.publish(msg)

    def set_value(self, value):
        self.value = str(value)

    def run(self):
        if not rclpy.ok():
            rclpy.init()
        self._spin_thread.start()

    def kill(self):
        self.destroy_node()


    def __spin(self):
        try:
            if not rclpy.ok():
                rclpy.init()
            rclpy.spin(self)
        except KeyboardInterrupt:
            self.destroy_node()
    

    def __del__(self):
        self.destroy_node()

def main():
    rclpy.init()
    node = sensor()
    # node2 = sensor('/hoge')
    # node2.set_value('hogehoge')
    node.run()
    # node2.run()
    node.set_value('first')
    time.sleep(3)
    node.set_value('second')
    time.sleep(10)

    
    node.kill()
    rclpy.shutdown()

if __name__=='__main__':
    main()