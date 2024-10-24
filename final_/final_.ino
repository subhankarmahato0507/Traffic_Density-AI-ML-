#define IRS1 A0
#define IRS2 A1
#define IRS3 A2
#define IRS4 A3
#define IRS5 A4
#define IRS6 A5
#define IRS7 2
#define IRS8 3

int IR1;
int IR2;
int IR3;
int IR4;
int IR5;
int IR6;
int IR7;
int IR8;

int a;
int b;
int c;
int d;


int LED1 = 4;
int LED2 = 5;
int LED3 = 6;
int LED4 = 7;
int LED5 = 8;
int LED6 = 9;
int LED7 = 10;
int LED8 = 11;

void setup() {
pinMode(IRS1, INPUT);
pinMode(IRS2, INPUT); 
pinMode(IRS3, INPUT);
pinMode(IRS4, INPUT); 
pinMode(IRS5, INPUT);
pinMode(IRS6, INPUT); 
pinMode(IRS7, INPUT);
pinMode(IRS8, INPUT); 

pinMode(LED1, OUTPUT); 
pinMode(LED2, OUTPUT); 
pinMode(LED3, OUTPUT); 
pinMode(LED4, OUTPUT); 
pinMode(LED5, OUTPUT); 
pinMode(LED6, OUTPUT); 
pinMode(LED7, OUTPUT); 
pinMode(LED8, OUTPUT); 

Serial.begin(9600);

}

void loop() 
{
  

  
 
irsensors();

if((a <=b) && (a <=c) && (a <=d)  )
{
 
A();
}
 irsensors();

if((b <=a) && (b <=c) && (b <=d)  )
   {
    B();
   }
   
irsensors();
if((c <=a) && ( c<=b) && (c <=d)  )
   {
 
C();
   }
irsensors();
if((d <=a) && (d <=b) && (d <=c)  )
   {
    D();
   }
  
}


void A()
{
  digitalWrite(LED1, HIGH);
  digitalWrite(LED3, LOW);
  digitalWrite(LED5, LOW);
  digitalWrite(LED7, LOW);
  digitalWrite(LED2, LOW);
  digitalWrite(LED4, HIGH);
  digitalWrite(LED6, HIGH);
  digitalWrite(LED8, HIGH);
  delay(5000);
  Serial.println(1);

}


void B()
{
  digitalWrite(LED1, LOW);
  digitalWrite(LED3, HIGH);
  digitalWrite(LED5, LOW);
  digitalWrite(LED7, LOW);
  digitalWrite(LED2, HIGH);
  digitalWrite(LED4, LOW);
  digitalWrite(LED6, HIGH);
  digitalWrite(LED8, HIGH);
  delay(5000);
  Serial.println(2);

}

void C()
{
  digitalWrite(LED1, LOW);
  digitalWrite(LED3, LOW);
  digitalWrite(LED5,HIGH);
  digitalWrite(LED7, LOW);
  digitalWrite(LED2, HIGH);
  digitalWrite(LED4, HIGH);
  digitalWrite(LED6, LOW);
  digitalWrite(LED8, HIGH);
  delay(5000);
  Serial.println(3);

}


void D()
{
  digitalWrite(LED1, LOW);
  digitalWrite(LED3, LOW);
  digitalWrite(LED5, LOW);
  digitalWrite(LED7, HIGH);
  digitalWrite(LED2, HIGH);
  digitalWrite(LED4, HIGH);
  digitalWrite(LED6, HIGH);
  digitalWrite(LED8, LOW);
  delay(5000);
  Serial.println(4);

}
void irsensors()
{
IR1 = digitalRead(IRS1);
IR2 = digitalRead(IRS2);
IR3 = digitalRead(IRS3);
IR4 = digitalRead(IRS4);
IR5 = digitalRead(IRS5);
IR6 = digitalRead(IRS6);
IR7 = digitalRead(IRS7);
IR8 = digitalRead(IRS8);
a=int(IR1)+int(IR2);
b=int(IR3)+int(IR4);
c=int(IR5)+int(IR6);
d=int(IR7)+int(IR8);
 //Serial.print("IR1 = ");
 Serial.print(IR1);
 Serial.print(","); 
 //Serial.print("IR2 = ");
 Serial.print(IR2);
Serial.print(","); 
//Serial.print("IR3 = ");
 Serial.print(IR3);
 Serial.print(","); 

//Serial.print("IR4 = ");
 Serial.print(IR4);
Serial.print(","); 
//Serial.print("IR5 = ");
 Serial.print(IR5);
Serial.print(","); 
//Serial.print("IR6 = ");
 Serial.print(IR6);
Serial.print(","); 
//Serial.print("IR7 = ");
Serial.print(IR7);
Serial.print(","); 
//Serial.print("IR8 = ");
Serial.print(IR8);
Serial.print(","); 

  
}
