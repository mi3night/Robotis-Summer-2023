import serial
import time


PORT_NAME = 'COM12'
ARDUINO_BAUD = 9600

arduino = serial.Serial(port=None, baudrate=ARDUINO_BAUD)

def serial_Start():
    arduino.port = PORT_NAME
    arduino.open()

def serial_Stop():
    arduino.close()    

def serial_WriteInt(mode):    
    time.sleep(2)
    print("Arduino Mode: ")
    print(str(mode).encode())
    arduino.write(str(mode).encode())

def serial_ReadInt():
    data = arduino.read()  
    # print(data) 
    return int.from_bytes(data, "big")


if __name__ == "__main__":
    serial_Start()
    serial_WriteInt(1)
    print(serial_ReadInt())
    serial_Stop()