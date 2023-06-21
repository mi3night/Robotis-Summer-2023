import arm_motions as motion
import serial_func as pod
import time

OPERATION_MODE = 1

if __name__ == "__main__":
    state = "REST"
    motion.start_Arm()
    dir_flag = True
    while (OPERATION_MODE):
        
        if state == "REST":
            motion.rest_Arm()
            mode = input("Enter 'Y' to start operation. Else, press any key: ")
            if (mode == 'Y'):
                print("Pod moving to one end of track in front of drone\n")
                pod.serial_Start()
                pod.serial_WriteInt(1)
                dest_data = pod.serial_ReadInt()
                print(dest_data)
                pod.serial_Stop()
                print("Wall detected. Arduino sending signals...\n")
                time.sleep(1)
                state = "DEPLETED_BAT_EXTRACTION" 
            else:    
                state = "STOP"
        elif state == "DEPLETED_BAT_EXTRACTION":
            motion.battery_Pull()
            print("Depleted battery pulled out and ressted on platform\n")
            time.sleep(1)
            state = "TRANSPORT"
        elif state == "TRANSPORT":
            
            pod.serial_Start()
            if (dest_data == 1):
                pod.serial_WriteInt(2)
                print("Pod moving to the other end of the track to BVM\n")
            elif (dest_data == 2):
                pod.serial_WriteInt(1)
                print("Pod moving to the other end of the track to drone\n")
            dest_data = pod.serial_ReadInt()
            print(dest_data)
            pod.serial_Stop()
            print("Wall detected. Arduino sending signals...\n")
            time.sleep(1)
            # direction = int(input("Enter pod moving direction: "))
            if (dest_data == 2):
                state = "DEPLETED_BAT_INSERTION"
            else:
                state = "FULL_BAT_INSERTION"
        elif state == "DEPLETED_BAT_INSERTION":
            motion.battery_Push()
            print("Depleted battery pushed into the chamber of BVM\n")
            time.sleep(1)
            state = "WAITING"
        elif state == "WAITING":
            time.sleep(10)     
            print("Waiting for BVM rotation for fully charged battery...\n")
            state = "FULL_BAT_EXTRACTION"
        elif state == "FULL_BAT_EXTRACTION":
            motion.battery_Pull()
            print("Fully charged battery pulled out and ressted on platform\n")
            time.sleep(1)
            state = "TRANSPORT"
        elif state == "FULL_BAT_INSERTION":
            motion.battery_Push()
            print("Fully charged battery pushed into the chamber of drone\n")
            time.sleep(1)
            state = "REST"   
        elif state == "STOP":
            OPERATION_MODE = 0
            motion.stop_Arm()         
        