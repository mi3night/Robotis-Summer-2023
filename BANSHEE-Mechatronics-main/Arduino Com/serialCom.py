
import motorctrl_v1 as motor
import serial
import time

arduino = serial.Serial('COM9', 1000000) 

time.sleep(2)  # wait for serial connection to establish

arduino.write(b'1')  # send '1' to turn on pod
time.sleep(5)
# arduino.write(b'0')  # send '0' to turn off pod
data = arduino.readline()
if data == '2':
    motor.portInitialization('COM3', [1])
    motor.dxlSetVelo([30],  [1])
    motor.motorRunWithInputs([225],[1])
    motor.portTermination()


arduino.write(b'0')  # send '1' to turn on pod
time.sleep(5)



#if(arm.claw==open && son1(drone) == 1)
#   do arm movement

