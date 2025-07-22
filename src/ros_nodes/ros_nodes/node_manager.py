from datetime import datetime
import threading
import time

import numpy as np
from termcolor import cprint, colored

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, String, Float32MultiArray
from rclpy.executors import SingleThreadedExecutor

def get_timestamp():
    return datetime.now().strftime('%Y%m%d_%H%M%S')


#センサー部
#トピック通信を行い、シミュレーターから値を送信する
#リストで通信する
class sensor_node_publisher(Node):
    '''
    ノードの名前を設定する
    timespan:何秒おきにデータをパブリッシュするか
    
    set_value'''
    def __init__(self, node_name, timespan=1.):
        self.nodename = str(node_name)
        super().__init__(self.nodename)
        self.value = []
        self.pub = self.create_publisher(msg_type=Float32MultiArray, 
                                         topic=self.nodename,
                                         qos_profile=5)
        self.timer = self.create_timer(timer_period_sec=timespan,
                                       callback=self.__timer_callback)
        self.lock = threading.Lock()
        self.get_logger().info(f'Publishing to topic: {self.nodename} every {timespan}s')


    def __timer_callback(self):
        with self.lock:
            msg = Float32MultiArray()
            msg.data = self.value
            self.pub.publish(msg)

    def set_value(self, value:list):
        value = value.copy()
        try:
            value = np.array(value, dtype=np.float32).tolist()
        except Exception as e:
            self.get_logger().error(f'Error in setting values: {e}')
        with self.lock:
            self.value = value


#センサーの受信
class sensor_node_subscriber(Node):
    '''
    sensor_nameに受信したいセンサーの名前を設定する
    read_sensor_valueで任意のタイミングでセンサーの値を読み取る
    '''

    def __init__(self, sensor_name):
        self.sensor_name = str(sensor_name)
        self.__value = []
        self.node_name = f'subs_{self.sensor_name}_{get_timestamp()}'
        super().__init__(self.node_name)
        self.subscription = self.create_subscription(msg_type=Float32MultiArray,
                                                      topic=self.sensor_name,
                                                      callback=self.__listener_callback,
                                                      qos_profile=10)
        self.lock = threading.Lock()

    def __listener_callback(self, msg):
        # self.get_logger().info('reciver value')
        value = msg.data
        with self.lock:
            self.__value = value

    def read_sensor_value(self):
        ret = None
        with self.lock:
                ret = self.__value
        return ret


#アクション部
#コントローラーから指示を受ける


#管理部
#多数のNodeをSingleThreadedExecutorで管理する
class NodeManager:
    '''
    add_sensor_node 送信用のノードを追加する
    add_resiver_node 受信用のノードを追加する(sensor_nameから受信する)

    '''
    def __init__(self):
        self.running = False
        rclpy.init()
        self.executor = SingleThreadedExecutor()
        self.dic_sensor_nodes_publisher = {}        #センサーの一覧を格納
        self.dic_sensor_nodes_subscriber = {}       #レシーバーの一覧を格納

    def add_sensor_node(self, node_name, timespan=1.):
        self.dic_sensor_nodes_publisher[node_name] = sensor_node_publisher(node_name, timespan)

    def add_reciver_node(self, sensor_name):
        self.dic_sensor_nodes_subscriber[sensor_name] = sensor_node_subscriber(sensor_name)
    

    def get_sensor_nodes_names(self):
        return list(self.dic_sensor_nodes_publisher.keys())

    def get_reciver_nodes_names(self):
        return list(self.dic_sensor_nodes_subscriber.keys())

    def set_sensor_node_value(self, node_name, value):
        if node_name in self.dic_sensor_nodes_publisher:
            self.dic_sensor_nodes_publisher[node_name].set_value(value)
        else:
            c_node_name = colored(node_name, color='yellow')
            cprint(f'no sensor node name: {c_node_name}')

    def get_reciver_node_value(self, sensor_name):
        ret = None
        if sensor_name in self.dic_sensor_nodes_subscriber:
            value = self.dic_sensor_nodes_subscriber[sensor_name].read_sensor_value()
            # print(f'sensor value {value}')
            ret = value
        else:
            c_node_name = colored(sensor_name, color='yellow')
            cprint(f'no sensor node name: {c_node_name}')
        return ret


    def run(self):
        if not self.running:
            self.running = True
            self.t = threading.Thread(target=self.__run_nodes)
            self.t.start()

    def __run_nodes(self):
        try:
            for node in self.dic_sensor_nodes_publisher.values():
                self.executor.add_node(node)
            for node in self.dic_sensor_nodes_subscriber.values():
                self.executor.add_node(node)
            self.executor.spin()
        finally:
            for node in self.dic_sensor_nodes_publisher.values():
                node.destroy_node()
            for node in self.dic_sensor_nodes_subscriber.values():
                node.destroy_node()
            self.executor.shutdown()
            rclpy.shutdown()

if __name__=='__main__':
    manager = NodeManager()
    manager.add_sensor_node('sensor1', timespan=0.5)
    manager.add_sensor_node('sensor2', timespan=1)
    manager.set_sensor_node_value('sensor1', [1,2,3])
    manager.set_sensor_node_value('sensor2', [4,5])
    manager.run()