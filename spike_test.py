import serial
import time

ser = serial.Serial('/dev/ttyACM0', 115200)

while True:
    msg = "Helo from PI!\n"
    ser.write(msg.encode())
    time.sleep(1)