import time
import ros_nodes



nManager = ros_nodes.node_manager.NodeManager()
nManager.add_reciver_node(sensor_name='sensor1')
nManager.add_reciver_node(sensor_name='sensor2')

nManager.run()
while True:
    print(nManager.get_reciver_node_value('sensor1'))
    print(nManager.get_reciver_node_value('sensor2'))
    time.sleep(3)
