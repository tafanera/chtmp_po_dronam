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
scan_results = []

detection_pub = rospy.Publisher('/color_scanner/detections', String, queue_size=10)
set_effect = rospy.ServiceProxy('led/set_effect', SetLEDEffect, persistent=True)
get_telemetry = rospy.ServiceProxy('get_telemetry', srv.GetTelemetry)
image_pub = rospy.Publisher('/Topic_Scan_Stepichev_Ilya', Image, queue_size=10)
result_pub = rospy.Publisher('/color_scanner/result', String, queue_size=1)
navigate = rospy.ServiceProxy('navigate', srv.Navigate)

last_detected_color = None

SCAN_RADIUS_PX = 90

TARGET_COLORS = {
    'red': ([0, 100, 100], [10, 255, 255], (255, 0, 0)),
    'green': ([40, 50, 50], [80, 255, 255], (0, 255, 0)),
    'blue': ([100, 100, 50], [140, 255, 255], (0, 0, 255))
}

def detect_dominant_color(roi):
    if roi.size == 0:
        return None, None

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    max_area = 0
    dominant_color = None
    dominant_rgb = None

    for color_name, (lower, upper, rgb) in TARGET_COLORS.items():
        lower = np.array(lower)
        upper = np.array(upper)
        mask = cv2.inRange(hsv, lower, upper)
        area = cv2.countNonZero(mask)
        if area > max_area:
            max_area = area
            dominant_color = color_name
            dominant_rgb = rgb

    return (dominant_color, dominant_rgb) if max_area > 50 else (None, None)


def image_callback(msg):
    global last_detected_color
    img = bridge.imgmsg_to_cv2(msg, 'bgr8')
    h, w = img.shape[:2]
    center_x, center_y = w // 2, h // 2

    mask_scan = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask_scan, (center_x, center_y), SCAN_RADIUS_PX, 255, -1)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    combined_mask = np.zeros((h, w), dtype=np.uint8)

    for color_name, (lower, upper, _) in TARGET_COLORS.items():
        lower = np.array(lower)
        upper = np.array(upper)
        color_mask = cv2.inRange(hsv, lower, upper)
        combined_mask = cv2.bitwise_or(combined_mask, cv2.bitwise_and(color_mask, mask_scan))

    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    detected_objects = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 300 or area > (SCAN_RADIUS_PX ** 2) * 3.14 * 0.7:
            continue

        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.035 * peri, True)

        if len(approx) == 3:
            shape = "Triangle"
        elif len(approx) == 4:
            x_box, y_box, w_box, h_box = cv2.boundingRect(approx)
            ratio = w_box / float(h_box)
            shape = "Square" if 0.85 <= ratio <= 1.15 else "Rectangle"
        elif len(approx) >= 5 and len(approx) <= 8:
            circularity = 4 * np.pi * area / (peri * peri)
            shape = "Circle" if circularity > 0.8 else "Polygon"
        else:
            circularity = 4 * np.pi * area / (peri * peri)
            shape = "Circle" if circularity > 0.75 else "Polygon"

        x, y, w_cnt, h_cnt = cv2.boundingRect(cnt)
        roi = img[y:y + h_cnt, x:x + w_cnt]
        color, rgb = detect_dominant_color(roi)

        if color:
            detected_objects.append((color, shape, rgb, cnt, (x, y)))
            cv2.drawContours(img, [cnt], -1, (255, 0, 255), 2)
            label = f"{color} {shape}"
            cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    if detected_objects:
        color, shape, rgb, _, _ = detected_objects[0]
        last_detected_color = color
        result_text = f"{color} {shape}"
        cv2.putText(img, f"Detected: {result_text}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        result_pub.publish(result_text)
        set_effect(r=rgb[2], g=rgb[1], b=rgb[0], effect='fill')
    else:
        last_detected_color = None
        set_effect(r=0, g=0, b=0, effect='fill')

    cv2.circle(img, (center_x, center_y), SCAN_RADIUS_PX, (0, 255, 0), 2)

    image_pub.publish(bridge.cv2_to_imgmsg(img, 'bgr8'))

rospy.Subscriber('/main_camera/image_raw_throttled', Image, image_callback, queue_size=1)


def scan_at_position(position_number):
    try:
        result_msg = rospy.wait_for_message('/color_scanner/result', String, timeout=3.0)
        result_text = result_msg.data.strip().lower()

        if result_text and result_text != "none none":
            scan_results.append(result_text)
            rospy.loginfo(f"object #{position_number}: {result_text}")

            if 'red' in result_text:
                set_effect(r=255, g=0, b=0, effect='blink')
            elif 'green' in result_text:
                set_effect(r=0, g=255, b=0, effect='blink')
            elif 'blue' in result_text:
                set_effect(r=0, g=0, b=255, effect='blink')
            rospy.sleep(1.0)
            set_effect(r=0, g=0, b=0, effect='fill')
        else:
            scan_results.append("not found")
            rospy.logwarn(f"object #{position_number} not found")
            set_effect(r=255, g=255, b=0, effect='blink')
            rospy.sleep(1.0)
            set_effect(r=0, g=0, b=0, effect='fill')

    except rospy.ROSException:
        scan_results.append("not found)")
        rospy.logwarn(f"trolllo #{position_number}")
        set_effect(r=255, g=0, b=0, effect='blink')
        rospy.sleep(1.0)
        set_effect(r=0, g=0, b=0, effect='fill')

def led_polovina():
    global last_detected_color

    if last_detected_color is None:
        rospy.logwarn("color not found.")
        set_effect(effect='fill', r=0, g=0, b=0)
        return

    set_effect(effect='fill', r=0, g=0, b=0)

    leds = []
    for i in range(72):
        if last_detected_color == 'Blue':
            if i >= 38:
                leds.append(LEDState(index=i, r=0, g=255, b=0))
            elif i in (36, 37):
                leds.append(LEDState(index=i, r=255, g=0, b=0))
            else:
                leds.append(LEDState(index=i, r=0, g=0, b=255))

        elif last_detected_color == 'Red':
            if i >= 38:
                leds.append(LEDState(index=i, r=0, g=0, b=255))
            elif i in (36, 37):
                leds.append(LEDState(index=i, r=255, g=255, b=255))
            else:
                leds.append(LEDState(index=i, r=255, g=0, b=0))

        elif last_detected_color == 'Green':
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

def perform_flight_route():
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
    scan_at_position(1)

    navigate(x=2, y=1, z=0.65, frame_id='aruco_map', speed=0.5)
    rospy.sleep(5)

    rospy.loginfo(f"Color: {last_detected_color}, Type: {type(last_detected_color)} ")
    navigate(x=2.5, y=1.5, z=0.65, frame_id='aruco_map', speed=0.5)
    rospy.sleep(10)
    led_polovina()
    scan_at_position(2)

    navigate(x=2.5, y=2.25, z=0.65, frame_id='aruco_map', speed=0.5)
    rospy.sleep(5)

    navigate(x=1, y=2, z=0.65, frame_id='aruco_map', speed=0.5)
    rospy.sleep(5)

    navigate(x=0, y=2, z=0.65, frame_id='aruco_map', speed=0.5)
    rospy.sleep(5)

    rospy.loginfo(f"Color: {last_detected_color}, Type: {type(last_detected_color)} ")
    navigate(x=0.5, y=1.5, z=0.65, frame_id='aruco_map', speed=0.5)
    rospy.sleep(10)
    led_polovina()
    scan_at_position(3)

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
        perform_flight_route()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr(f"error: {e}")
    finally:
        try:
            set_effect(r=0, g=0, b=0, effect='fill')
        except Exception:
            pass

        try:
            with open('otchet_Stepichev_Ilya.txt', "w", encoding="utf-8") as f:
                if scan_results:
                    for i, obj in enumerate(scan_results, 1):
                        f.write(f"{i}. {obj.capitalize()}\n")
                else:
                    f.write("object not found\n")
        except Exception as e:
            rospy.logerr(f"error: {e}")