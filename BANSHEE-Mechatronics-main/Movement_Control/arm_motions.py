import time
import keyboard
import motorctrl_v1 as motor
import Movement_Calc_v2 as calculation


BASE_ID = 1
BICEP_ID = 2
FOREARM_ID = 3
WRIST_ID = 4
CLAW_ID = 0

PORT_NUM = 'COM3'
BAUDRATE = 1000000

ALL_IDs = [BASE_ID, BICEP_ID, FOREARM_ID, WRIST_ID, CLAW_ID]
MOVE_IDs = [BASE_ID, BICEP_ID, FOREARM_ID, WRIST_ID]
# COOR_1 = [5, -275, 40]                                      # In front of the chamber
# COOR_2 = [5, -225, 40]                                      # Away from chamber   

# COOR_1 = [285, 0, 40]                                     
COOR_IN1 = [225, 0, 40]                                       # In front of the chamber    
COOR_IN2 = [225, 0, 60]         
# COOR_4 = [245, 0, 40]  
COOR_OUT = [375, 0, 70]                                       # Away from chamber   
# COOR_2 = [225, 0, 40]                                      

COOR_REST = [275, 0, 205]  

def start_Arm():
    motor.portInitialization(PORT_NUM, ALL_IDs)
    print("DXL PORTS INITIALIZED \n")

def stop_Arm():
    motor.portTermination()
    print("DXL PORTS TERMINATED \n")

def battery_Pull():
    motor.dxlSetVelo([30,18,60,50,30], ALL_IDs)

    motor.motorRunWithInputs([90], [CLAW_ID])               # Open the claws

    angles = calculation.angle_Calc(COOR_OUT, 0)              # End effector positioned to the front of battery # Forearm always parallel to ground <-> 0 for Forearm Mode
    motor.simMotorRun(angles, MOVE_IDs)                     

    motor.motorRunWithInputs([180], [CLAW_ID])              # Close the claws

    angles = calculation.angle_Calc(COOR_IN2, 0)              # End effector pulled away from the chamber
    motor.simMotorRun(angles, MOVE_IDs)
    print("BATTERY PULLED\n")



                     
def battery_Push():
    # motor.dxlSetVelo([30,18,30,30,60], ALL_IDs)
    motor.dxlSetVelo([30,15,50,30,30], ALL_IDs)

    motor.motorRunWithInputs([180], [CLAW_ID])              # Close the claws

    angles = calculation.angle_Calc(COOR_IN1, 0)              # End effector away from the chamber # Forearm always parallel to ground <-> 0 for Forearm Mode
    motor.simMotorRun(angles, MOVE_IDs)                     

    angles = calculation.angle_Calc(COOR_OUT, 0)              # End effector push battery into the chamber
    motor.simMotorRun(angles, MOVE_IDs)

    motor.motorRunWithInputs([90], [CLAW_ID])               # Open the claws
    print("BATTERY PUSHED\n")


def rest_Arm():
    motor.dxlSetVelo([30,18,30,30,30], ALL_IDs)

    angles = calculation.angle_Calc(COOR_REST, 0)
    motor.simMotorRun(angles, MOVE_IDs)  
    print("ARM AT REST\n")



if __name__ == "__main__":
    while True:  # making a loop
        try:  # used try so that if user pressed other than the given key error will not be shown
            if keyboard.is_pressed('0'):  # if key '0' is pressed 
                start_Arm()               # finishing the loop
            if keyboard.is_pressed('1'):  # if key '1' is pressed 
                battery_Push()            # finishing the loop
            elif keyboard.is_pressed('2'):  # if key 01'2' is pressed 
                battery_Pull()              # finishing the loop
            elif keyboard.is_pressed('3'):  # if key '3' is pressed 
                rest_Arm()                  # finishing the loop
            elif keyboard.is_pressed('4'):
                stop_Arm()
                break
        except:
            stop_Arm()
            break 
    # start_Arm()
    # time.sleep(2)
    # # battery_Pull()
    # # time.sleep(2)
    # # battery_Push()
    # # time.sleep(2)
    # # motor.dxlSetVelo([30,15,50,30,300], ALL_IDs)
    # # angles = calculation.angle_Calc(COOR_3, 0)              # End effector positioned to the front of battery # Forearm always parallel to ground <-> 0 for Forearm Mode
    # # motor.simMotorRun(angles, MOVE_IDs) 
    # # time.sleep(5)  

    # rest_Arm()010
    # time.sleep(5)
    # stop_Arm()