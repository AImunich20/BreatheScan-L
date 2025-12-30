// Pin definitions
#define MQ_135 A0  // Acetone
#define MQ_136 A1  // H2S
#define MQ_137 A2  // Ammonia
#define MQ_138 A3  // Formaldehyde

// Variables for sensor readings
int sensorValue135 = 0;
int sensorValue136 = 0;
int sensorValue137 = 0;
int sensorValue138 = 0;

float Rs; // Sensor resistance for MQ-136

// Function prototype
float analyzeH2S(int adc);

void setup() {
  Serial.begin(9600);
}

void loop() {
  // Read MQ-135 (Acetone)
  sensorValue135 = analogRead(MQ_135);
  Serial.print("Acetone ADC: ");
  Serial.println(sensorValue135);

  // Read MQ-136 (H2S)
  sensorValue136 = analogRead(MQ_136);
  Serial.print("H2S ADC: ");
  Serial.print(sensorValue136);
  Serial.print("\t");

  float h2sPPM = analyzeH2S(sensorValue136);
  Serial.print("H2S: ");
  Serial.print(h2sPPM, 3);
  Serial.print(" ppm\t");

  Serial.print("Sensor Resistance (Rs): ");
  Serial.println(Rs, 2);

  // Read MQ-137 (Ammonia)
  sensorValue137 = analogRead(MQ_137);
  Serial.print("Ammonia ADC: ");
  Serial.println(sensorValue137);

  // Read MQ-138 (Formaldehyde)
  sensorValue138 = analogRead(MQ_138);
  Serial.print("Formaldehyde ADC: ");
  Serial.println(sensorValue138);

  Serial.println("-------------------------------------");
  delay(500);
}

// Function to calculate H2S concentration from MQ-136 ADC value
float analyzeH2S(int adc) {
  const float slope = -0.4150374993;
  const float A = 5.409597909;
  const float Rseries = 1000.0;
  const float Vadc = (adc * 5.0) / 1023.0;

  Rs = ((5.0 - Vadc) / Vadc) * Rseries;

  const float R0 = 1966.78;
  float ratio = Rs / R0;

  float h2sPPM = pow(10, (log10(ratio / A) / slope));
  return h2sPPM;
}
