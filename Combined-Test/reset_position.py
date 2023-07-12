import math
import motorctrl_v1 as motor
import Movement_Calc_v2 as calculation
import numpy as np
import time
import time
import cv2


BASE_ID = 1
BICEP_ID = 2
FOREARM_ID = 3
WRIST_ID = 4
CLAW_ID = 0

PORT_NUM = 'COM5'
BAUDRATE = 1000000

MOVEARM_MODE = 1

ALL_IDs = [BASE_ID, BICEP_ID, FOREARM_ID, WRIST_ID, CLAW_ID]
MOVE_IDs = [BASE_ID, BICEP_ID, FOREARM_ID, WRIST_ID, CLAW_ID]

frameX = 0
objX = 0
frameY = 0
objY = 0

motor.portInitialization(PORT_NUM, ALL_IDs)
motor.dxlSetVelo([20, 20, 20, 20, 20],[0, 1, 2, 3, 4])
motor.motorRunWithInputs([180], [4])
motor.motorRunWithInputs([90, 227, 273, 47], [0, 1, 2, 3])
time.sleep(1)
current2 = 273
current3 = 47
current4 = 180
for i in range(8):
    current2 -= 90/8
    current3 += 190/8.5
    current4 -= 87/8.5
    motor.motorRunWithInputs([90, 227, current2, current3, current4], [0, 1, 2, 3, 4])
    time.sleep(1)
