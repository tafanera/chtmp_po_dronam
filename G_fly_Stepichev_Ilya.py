import rospy
from clover.srv import Navigate
from std_srvs.srv import Trigger
from clover.srv import SetLEDEffect
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import String
import numpy as np
import cv2
from pyzbar import pyzbar
from clover import srv
import math
from led_msgs.srv import SetLEDs
from led_msgs.msg import LEDState
import random



rospy.init_node('flight', anonymous=True)
bridge = CvBridge()
kernel = np.ones((5, 5), np.uint8)

colors_hsv = {
    "Red": [(np.array([0, 62, 80]), np.array([17, 197, 162]))],
    "Green": [(np.array([34, 131, 0]), np.array([73, 255, 197]))],
    "Blue": [(np.array([96, 103, 56]), np.array([113, 255, 104]))]
}

image_pub = rospy.Publisher('/Topic_Scan_Stepichev_Ilya', Image, queue_size=10)
detection_pub = rospy.Publisher('/color_scanner/detections', String, queue_size=10)
set_effect = rospy.ServiceProxy('led/set_effect', SetLEDEffect, persistent=True)
get_telemetry = rospy.ServiceProxy('get_telemetry', srv.GetTelemetry)

navigate = rospy.ServiceProxy('navigate', srv.Navigate)

last_detected_color = None


def navigate_wait(x=0, y=0, z=0, speed=0.5, frame_id='body', auto_arm=False):
    res = navigate(x=x, y=y, z=z, speed=speed, frame_id=frame_id, auto_arm=auto_arm, yaw=float('nan'))

    if not res.success:
        raise Exception(res.message)

    while not rospy.is_shutdown():
        telem = get_telemetry(frame_id='navigate_target')
        if math.sqrt(telem.x ** 2 + telem.y ** 2 + telem.z ** 2) < 0.2:
            return
        rospy.sleep(0.2)


def image_callback(data):
    global last_detected_color
    try:
        image = bridge.imgmsg_to_cv2(data, 'bgr8')
    except Exception as e:
        rospy.logerr_throttle(5, f"Ошибка конвертации: {e}")
        return
    if image.size == 0:
        return

    output = image.copy()
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, w = image.shape[:2]
    image_area = h * w
    detections = []

    for color_name, ranges in colors_hsv.items():
        mask = np.zeros((h, w), dtype=np.uint8)
        for lower, upper in ranges:
            mask |= cv2.inRange(hsv, lower, upper)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 500 or area > image_area * 0.9:
                continue
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
            if len(approx) == 3:
                shape = "Triangle"
            elif len(approx) == 4:
                x, y, w_box, h_box = cv2.boundingRect(approx)
                ratio = w_box / float(h_box)
                shape = "Square" if 0.9 <= ratio <= 1.1 else "Rectangle"
            elif len(approx) > 4:
                shape = "Circle"
            else:
                continue

            last_detected_color = color_name

            cv2.drawContours(output, [cnt], -1, (0, 255, 0), 2)
            x, y, w_box, h_box = cv2.boundingRect(cnt)
            label = f"{color_name} {shape}"
            cv2.putText(output, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            detections.append(f"{color_name} {shape}")

    image_pub.publish(bridge.cv2_to_imgmsg(output, 'bgr8'))
    if detections:
        detection_pub.publish("; ".join(detections))
        rospy.loginfo_throttle(2, f"scan: {'; '.join(detections)}")




rospy.Subscriber('/main_camera/image_raw_throttled', Image, image_callback, queue_size=1)
def led_polovina():
    global last_detected_color

    if last_detected_color is None:
        rospy.logwarn("Цвет не обнаружен! Светодиоды выключены.")
        set_effect(effect='fill', r=0, g=0, b=0)
        return

    # Сначала гасим всё, чтобы эффект был понятнее
    set_effect(effect='fill', r=0, g=0, b=0)

    leds = []
    for i in range(72):
        if last_detected_color == 'Blue':
            # пример: правая часть зелёная, 2 диода разделителя красные
            if i >= 38:
                leds.append(LEDState(index=i, r=0, g=255, b=0))
            elif i in (36, 37):
                leds.append(LEDState(index=i, r=255, g=0, b=0))
            else:
                leds.append(LEDState(index=i, r=0, g=0, b=255))

        elif last_detected_color == 'Red':
            # пример: правая часть синяя, 2 диода белые
            if i >= 38:
                leds.append(LEDState(index=i, r=0, g=0, b=255))
            elif i in (36, 37):
                leds.append(LEDState(index=i, r=255, g=255, b=255))
            else:
                leds.append(LEDState(index=i, r=255, g=0, b=0))

        elif last_detected_color == 'Green':
            # рандом на каждый диод
            leds.append(LEDState(index=i,
                                 r=random.randint(0, 255),
                                 g=random.randint(0, 255),
                                 b=random.randint(0, 255)))
        else:
            leds.append(LEDState(index=i, r=0, g=0, b=0))

    try:
        set_leds(leds)
    except Exception as e:
        rospy.logerr(f"led/set_leds failed: {e}")


set_leds = rospy.ServiceProxy('led/set_leds', SetLEDs, persistent=True)


def polet():
    rospy.wait_for_service('navigate')

    rospy.loginfo("fly...")
    navigate(x=0, y=0, z=0.65, frame_id='body', auto_arm=True)
    led_polovina()
    rospy.sleep(10)

    navigate(x=1, y=0, z=0.65, frame_id='aruco_map', speed=0.5)
    rospy.sleep(5)

    rospy.loginfo(f"Color: {last_detected_color}, Type: {type(last_detected_color)} ")
    navigate(x=1.5, y=0.5, z=0.65, frame_id='aruco_map', speed=0.5)
    rospy.sleep(10)
    led_polovina()

    navigate(x=2, y=1, z=0.65, frame_id='aruco_map', speed=0.5)
    rospy.sleep(5)

    rospy.loginfo(f"Color: {last_detected_color}, Type: {type(last_detected_color)} ")
    navigate(x=2.5, y=1.5, z=0.65, frame_id='aruco_map', speed=0.5)
    rospy.sleep(10)
    set_effect(effect='rainbow')
    led_polovina()

    navigate(x=2.5, y=2.25, z=0.65, frame_id='aruco_map', speed=0.5)
    rospy.sleep(5)

    navigate(x=1, y=2, z=0.65, frame_id='aruco_map', speed=0.5)
    rospy.sleep(5)

    navigate(x=0, y=2, z=0.65, frame_id='aruco_map', speed=0.5)
    rospy.sleep(5)

    rospy.loginfo(f"Color: {last_detected_color}, Type: {type(last_detected_color)} ")
    navigate(x=0.5, y=1.5, z=0.2, frame_id='aruco_map', speed=0.5)
    rospy.sleep(10)
    led_polovina()

    navigate(x=0, y=1, z=0.65, frame_id='aruco_map', speed=0.5)
    rospy.sleep(5)

    navigate(x=0, y=0, z=0.65, frame_id='aruco_map', speed=0.5)
    rospy.sleep(5)



    rospy.wait_for_service('land')
    rospy.ServiceProxy('land', Trigger)()
    set_effect(r=0, g=0, b=0, effect='fill')
    rospy.loginfo("posadka")



if __name__ == '__main__':
    try:
        polet()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr(f"Ошибка: {e}")
    finally:
        try:
            set_effect(r=0, g=0, b=0, effect='fill')
        except Exception:
            pass
        try:
            with open('otchet_Stepichev_Ilya.txt', "w", encoding="utf-8") as f:
                f.write(str(last_detected_color))
        except Exception as e:
            rospy.logerr(f"Не смог записать отчёт: {e}")

