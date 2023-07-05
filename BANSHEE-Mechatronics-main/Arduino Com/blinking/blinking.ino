void setup() {
  pinMode(6, OUTPUT);
  digitalWrite(6, LOW);
  Serial.begin(9600);
}

void loop() {
  if (Serial.available()>0) {
//    char incomingByte = Serial.read();
    int mode = Serial.parseInt();
//    Serial.print("I received: ");
//    Serial.println(incomingByte);
//    if (incomingByte == '1') {
//      digitalWrite(LED_BUILTIN, HIGH);
//      Serial.write('2');
//    }
//    else if (incomingByte == '0') {
//      digitalWrite(LED_BUILTIN, LOW);
//    }

    switch(mode) {
      case 1:
        digitalWrite(6, HIGH);
        Serial.write(2);
        break;
      case 2:
        digitalWrite(6, LOW);
        break;
      default:
        break;
    }
  }
}
