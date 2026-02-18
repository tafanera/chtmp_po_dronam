карта в газебу накатить


sed -i 's|arg name="aruco_vpe" default="false"|arg name="aruco_vpe" default="true"|g' /home/clover/catkin_ws/src/clover/clover/launch/aruco.launch
sed -i 's|arg name="length" default="0.22"|arg name="length" default="0.33"|g' /home/clover/catkin_ws/src/clover/clover/launch/aruco.launch
sed -i 's|arg name="map" default="map.txt"|arg name="map" default="guap.txt"|g' /home/clover/catkin_ws/src/clover/clover/launch/aruco.launch
sed -i 's|arg name="aruco" default="false"|arg name="aruco" default="true"|g' /home/clover/catkin_ws/src/clover/clover/launch/clover.launch

rosrun aruco_pose genmap.py 0.33 4 4 1 1 0 -o guap.txt

find ./ -name 'guap.txt'


cat ./catkin_ws/src/clover/aruco_pose/map/guap.txt

rosrun clover_simulation aruco_gen --single-model --source-world=/home/clover/catkin_ws/src/clover/clover_simulation/resources/worlds/clover.world /home/clover/catkin_ws/src/clover/aruco_pose/map/guap.txt > /home/clover/catkin_ws/src/clover/clover_simulation/resources/worlds/guap.world

sed -i 's|clover_aruco.world|guap.world|g' /home/clover/catkin_ws/src/clover/clover_simulation/launch/simulator.launch
