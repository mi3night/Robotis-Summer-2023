const int L_led = 11;
const int R_led = 12;
 
const int pwm1 = 6; // pin 10 as pwm
const int dir1 = 5;  // pin 9 as dir
const int pwm2 = 10; // pin 10 as pwm
const int dir2 = 9;  // pin 9 as dir
 //schamtic pins 
// const int trig1Pin = 7;   //7 <- schematic pin right // test port: 4
// const int echo1Pin = 8; //8 <-schematic pin right // test port 3
// const int trig2Pin = 4; //4 <-schematic pin left // test port: 8
// const int echo2Pin = 3; //3 <- shcematic pin left // test port: 7

//test pins 
const int trig1Pin = 4;   //7 <- schematic pin right // test port: 4
const int echo1Pin = 3; //8 <-schematic pin right // test port 3
const int trig2Pin = 8; //4 <-schematic pin left // test port: 8
const int echo2Pin = 7; //3 <- shcematic pin left // test port: 7

int buff = 0;

bool cur_dir = true;
float duration;

unsigned int distance1;
unsigned int distance2;

enum {STOP, RIGHT_MOVE, LEFT_MOVE};
unsigned char state;

float distanceDetect(int trigPin, int echoPin){
  //shoot out ultrasonic wave
  digitalWrite(trigPin, LOW);
  delayMicroseconds(2);
  digitalWrite(trigPin, HIGH);
  delayMicroseconds(10);
  digitalWrite(trigPin, LOW);

  //detect the time during which the ultrasonic wave travels
  duration = pulseIn(echoPin, HIGH);
  //calculate the distance from the obstacle using the speed of sound
  float range = (duration*.0343)/2;
//  Serial.print("Distance: ");
//  Serial.println(range);   
  delay(50);
  return range;
}

void blinkLED(){
  digitalWrite(6, HIGH);
  delay(200);
  digitalWrite(6, LOW);
  delay(200);
  digitalWrite(6, HIGH);
  delay(200);
}
 
void setup(){

  Serial.begin(9600);

  pinMode(L_led, OUTPUT);
  pinMode(R_led, OUTPUT);
  
  pinMode(pwm1,OUTPUT); //declare pin pwm as OUTPUT
  pinMode(dir1,OUTPUT); //declare pin dir as OUTPUT
  pinMode(pwm2,OUTPUT); //declare pin pwm as OUTPUT
  pinMode(dir2,OUTPUT); //declare pin dir as OUTPUT

  pinMode(echo1Pin, INPUT);  
  pinMode(trig1Pin, OUTPUT);
  pinMode(echo2Pin, INPUT);  
  pinMode(trig2Pin, OUTPUT);
      
  digitalWrite(dir1,HIGH);  // if DIR pin is HIGH, B will HIGH ; if DIR pin is LOW, A will HIGH
  digitalWrite(dir2,HIGH);  // if DIR pin is HIGH, B will HIGH ; if DIR pin is LOW, A will HIGH

  analogWrite(pwm1, 0);
  analogWrite(pwm2, 0);  
  
  digitalWrite(L_led,LOW);
  digitalWrite(R_led,LOW);
}

void loop() {
  // put your main code here, to run repeatedly:
//  Serial.print("Sensor 1 distance: ");
  distance1 = distanceDetect(trig1Pin, echo1Pin);
//  Serial.print("Sensor 2 distance: ");
  distance2 = distanceDetect(trig2Pin, echo2Pin);
  delay(50);

  if (Serial.available()>0) {
    int mode = Serial.parseInt();

    switch(mode) {
      case 1:                       //Mode 1: Pod moving to the right
        digitalWrite(6, HIGH);
//        Serial.write(2);
        state = RIGHT_MOVE;        
        break;
      case 2:                       //Mode 2: Pod moving to the left
        digitalWrite(6, LOW);
        state = LEFT_MOVE;
        break;
      default:
        break;
    }
  }

  if (state == RIGHT_MOVE) {
//        Serial.println("Moving right");
        // Turning on LED indicating direction RIGHT
        digitalWrite(L_led,LOW);
        digitalWrite(R_led,HIGH);        
        // Change direction of motors
        cur_dir=true;              // false -> left
        digitalWrite(dir1,cur_dir);
        digitalWrite(dir2,cur_dir);
        //Start driving the motors
        analogWrite(pwm1, 155);
        analogWrite(pwm2, 155);
        if (distance1<15 && cur_dir)  {
          state = STOP;
          Serial.write(1);
        }
  }
  else if (state == LEFT_MOVE) {
//        Serial.println("Moving left");
        // Turning on LED indicating direction LEFT
        digitalWrite(L_led,HIGH);
        digitalWrite(R_led,LOW);        
        // Change direction of motors
        cur_dir=false;              // false -> left
        digitalWrite(dir1,cur_dir);
        digitalWrite(dir2,cur_dir);
        //Start driving the motors
        analogWrite(pwm1, 155);
        analogWrite(pwm2, 155);
        if (distance2<15 && !cur_dir)  {
          state = STOP;
          Serial.write(2);
        }   
  }
  else if (state == STOP) {
//        Serial.println("Stop");
//        digitalWrite(6, LOW);
       analogWrite(pwm1, 0);
       analogWrite(pwm2, 0); 
       digitalWrite(L_led,LOW);
       digitalWrite(R_led,LOW);   
  } 
}
