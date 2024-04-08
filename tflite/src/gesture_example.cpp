
/*
 ---- WHAT IS IT? ------
 A Gesture training ML Example for ESP32 based on the Arduino Article:
 https://blog.arduino.cc/2019/10/15/get-started-with-machine-learning-on-arduino/
 Running on ESP32 with BMA423 Accellerometer, as part of TTGO-WATCH-2020 here.

 ---- HOW TO USE THIS EXAMPLE ------
 1) Configure the WIFI Credentials for your network
 2) Upload with ALREADYTRAINED Off via Serial, with Serial Debug and UDP
 Transport options set 3) Can disconnect USB Here from device 4) Record same
 action a number of times, it pauses for 5s after each one before re-enabling
 the motion sensor 5) Save the Output from the Serial Monitor into the text
 files e.g. flex.csv and punch.csv 6) Build Model in Colab:
 https://colab.research.google.com/drive/18DG8Ld6JwumUAIAlw96CMvnRSATY-wiU 7)
 Download Model.h to this project and replace the empty one it comes with 8)
 Uncomment ALREADYTRAINED and Build and upload to your board 9) Now it will
 display what action it thinks has happened via the Serial Monitor! (and on
 screen)
*/

// Essentially a Mashup of:
// TTGO Watch BMA423_Accel Example
// TensorFlow ESP32 Hello World Example
// TensorFlow Arduino Example for Gesture Training with Google Colab Model
// Generation ESP32 OTA Upload Example vMicro Serial Debugger via OTA +
// SendUserMessage feature

// Add this in once you have trained and downloaded the Tensorflow Model....

#include "../include/global.h"

#define ALREADYTRAINED

#ifdef ALREADYTRAINED
// #include <TensorFlowLite_ESP32.h>

// #include "D:\\BK\\Thesis\\Source\\source\\Lilygo\\testBMA\\include\\g_model.h"
#include "../include/model/g_model.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace TFModel
{  // global variables used for TensorFlow Lite (Micro)
    tflite::ErrorReporter *tflErrorReporter = NULL;

    // pull in all the TFLM ops, you can remove this line and
    // only pull in the TFLM ops you need, if would like to reduce
    // the compiled size of the sketch.
    static tflite::AllOpsResolver resolver;

    const tflite::Model *tflModel = nullptr;
    tflite::MicroInterpreter *tflInterpreter = nullptr;
    TfLiteTensor *tflInputTensor = nullptr;
    TfLiteTensor *tflOutputTensor = nullptr;

    // Create a static memory buffer for TFLM, the size may need to
    // be adjusted based on the model you are using
    constexpr int tensorArenaSize = 8 * 1024;
    byte tensorArena[tensorArenaSize];
}  // namespace TFModel

// array to map gesture index to a name
const char *GESTURES[] = {"punch", "flex"};

// #define NUM_GESTURES (sizeof(GESTURES) / sizeof(GESTURES[0]))
#define NUM_GESTURES 2

#endif

TTGOClass *watch;
TFT_eSPI *tft;
BMA *sensor;

void setup()
{
    Serial.begin(115200);

    // Get TTGOClass instance
    watch = TTGOClass::getWatch();

    // Initialize the hardware, the BMA423 sensor has been initialized
    // internally
    watch->begin();

    // Turn on the backlight
    watch->openBL();

    // Receive objects for easy writing
    tft = watch->tft;
    sensor = watch->bma;

    // Accel parameter structure
    Acfg cfg;
    /*!
        Output data rate in Hz, Optional parameters:
            - BMA4_OUTPUT_DATA_RATE_0_78HZ
            - BMA4_OUTPUT_DATA_RATE_1_56HZ
            - BMA4_OUTPUT_DATA_RATE_3_12HZ
            - BMA4_OUTPUT_DATA_RATE_6_25HZ
            - BMA4_OUTPUT_DATA_RATE_12_5HZ
            - BMA4_OUTPUT_DATA_RATE_25HZ
            - BMA4_OUTPUT_DATA_RATE_50HZ
            - BMA4_OUTPUT_DATA_RATE_100HZ
            - BMA4_OUTPUT_DATA_RATE_200HZ
            - BMA4_OUTPUT_DATA_RATE_400HZ
            - BMA4_OUTPUT_DATA_RATE_800HZ
            - BMA4_OUTPUT_DATA_RATE_1600HZ
    */
    cfg.odr = BMA4_OUTPUT_DATA_RATE_100HZ;
    /*!
        G-range, Optional parameters:
            - BMA4_ACCEL_RANGE_2G
            - BMA4_ACCEL_RANGE_4G
            - BMA4_ACCEL_RANGE_8G
            - BMA4_ACCEL_RANGE_16G
    */
    cfg.range = BMA4_ACCEL_RANGE_2G;
    /*!
        Bandwidth parameter, determines filter configuration, Optional
       parameters:
            - BMA4_ACCEL_OSR4_AVG1
            - BMA4_ACCEL_OSR2_AVG2
            - BMA4_ACCEL_NORMAL_AVG4
            - BMA4_ACCEL_CIC_AVG8
            - BMA4_ACCEL_RES_AVG16
            - BMA4_ACCEL_RES_AVG32
            - BMA4_ACCEL_RES_AVG64
            - BMA4_ACCEL_RES_AVG128
    */
    cfg.bandwidth = BMA4_ACCEL_NORMAL_AVG4;

    /*! Filter performance mode , Optional parameters:
        - BMA4_CIC_AVG_MODE
        - BMA4_CONTINUOUS_MODE
    */
    cfg.perf_mode = BMA4_CONTINUOUS_MODE;

    // Configure the BMA423 accelerometer
    sensor->accelConfig(cfg);

    // Enable BMA423 accelerometer
    sensor->enableAccel();

    // You can also turn it off
    // sensor->disableAccel();

    // Print Header for Data File Output
    Serial.println("aX,aY,aZ");
    // MicroDebug.sendUserMessage("aX,aY,aZ");

    // Some display settings
    tft->setTextColor(random(0xFFFF));
    tft->drawString("Waiting....", 25, 50, 4);

#ifdef ALREADYTRAINED
    static tflite::MicroErrorReporter micro_error_reporter;
    TFModel::tflErrorReporter = &micro_error_reporter;
    // get the TFL representation of the model byte array
    TFModel::tflModel = tflite::GetModel(g_model);
    if (TFModel::tflModel->version() != TFLITE_SCHEMA_VERSION)
    {
        Serial.println("Model schema mismatch!");
        while (1)
            ;
    }

    static tflite::MicroMutableOpResolver<7> micro_mutable_op_resolver;
    if (micro_mutable_op_resolver.AddFullyConnected() != kTfLiteOk)
    {
        TFModel::tflErrorReporter->Report("Wrong ops");
        while (1)
            ;
    }
    if (micro_mutable_op_resolver.AddExp() != kTfLiteOk)
    {
        TFModel::tflErrorReporter->Report("Wrong ops");
        while (1)
            ;
    }
    if (micro_mutable_op_resolver.AddAdd() != kTfLiteOk)
    {
        TFModel::tflErrorReporter->Report("Wrong ops");
        while (1)
            ;
    }
    if (micro_mutable_op_resolver.AddLog() != kTfLiteOk)
    {
        TFModel::tflErrorReporter->Report("Wrong ops");
        while (1)
            ;
    }
    if (micro_mutable_op_resolver.AddSoftmax() != kTfLiteOk)
    {
        TFModel::tflErrorReporter->Report("Wrong ops");
        while (1)
            ;
    }

    // Create an interpreter to run the model
    TFModel::tflInterpreter = new tflite::MicroInterpreter(TFModel::tflModel, micro_mutable_op_resolver, TFModel::tensorArena, TFModel::tensorArenaSize, TFModel::tflErrorReporter);

    // Allocate memory for the model's input and output tensors
    TFModel::tflInterpreter->AllocateTensors();
    TfLiteStatus allocate_status = TFModel::tflInterpreter->AllocateTensors();
    if (allocate_status != kTfLiteOk)
    {
        TFModel::tflErrorReporter->Report("AllocateTensors() failed!");
        while (1)
            ;
    }

    // Get pointers for the model's input and output tensors
    TFModel::tflInputTensor = TFModel::tflInterpreter->input(0);
    TFModel::tflOutputTensor = TFModel::tflInterpreter->output(0);
    Serial.print("Type: ");
    Serial.println(TFModel::tflInputTensor->type);
    Serial.print("Dim input: ");
    Serial.printf("%d %d\n", TFModel::tflInputTensor->dims->data[0], TFModel::tflInputTensor->dims->data[1]);
    Serial.print("Dim output: ");
    Serial.printf("%d %d\n", TFModel::tflOutputTensor->dims->data[0], TFModel::tflOutputTensor->dims->data[1]);
    Serial.printf("Buffer: %d %d\n", TFModel::tflInputTensor->data, TFModel::tflOutputTensor->data);
#endif
}

const float accelerationThreshold = 100;  // threshold of significant motion (per-axis - see line 265ish)
const int numSamples = 119;

int samplesRead = numSamples;
int outputCount = 0;

int16_t cxpos = 0;
int16_t cypos = 0;
int16_t czpos = 0;
unsigned long lastReadingDone = 0;
bool isWaiting = false;
void loop()
{
    Accel acc;
    // wait for significant motion
    while (samplesRead == numSamples)
    {
        // ArduinoOTA.handle();
        bool res = sensor->getAccel(acc);
        if (res)
        {
            // Show the data on screen for user
            tft->setTextFont(4);
            tft->setTextColor(TFT_WHITE, TFT_BLACK);
            tft->fillRect(98, 100, 70, 85, TFT_BLACK);
            tft->setCursor(80, 100);
            tft->print("X:");
            tft->println(acc.x);
            tft->setCursor(80, 130);
            tft->print("Y:");
            tft->println(acc.y);
            tft->setCursor(80, 160);
            tft->print("Z:");
            tft->println(acc.z);

            if ((millis() - lastReadingDone) > 5000)
            {
                // compare to last reading
                if ((abs(cxpos - acc.x) > accelerationThreshold) || (abs(cypos - acc.y) > accelerationThreshold) || (abs(czpos - acc.z) > accelerationThreshold))
                {
                    // reset the sample read count
                    samplesRead = 0;
                    tft->fillScreen(TFT_GREEN);
                    tft->setTextColor(TFT_GREEN, TFT_BLACK);
                    tft->drawString("Reading....", 25, 50, 4);
                    isWaiting = false;
                }
                else
                {
                    if (isWaiting == false)
                    {
                        isWaiting = true;
                        tft->drawString("Waiting....", 25, 50, 4);
                    }
                }
            }
            // read the acceleration data
            cxpos = acc.x;
            cypos = acc.y;
            czpos = acc.z;
        }
    }

    // check if the all the required samples have been read since
    // the last time the significant motion was detected
    while (samplesRead < numSamples)
    {
        bool res = sensor->getAccel(acc);
        // check if both new acceleration and gyroscope data is
        // available
        if (res)
        {
            // read the acceleration and gyroscope data
            cxpos = acc.x;
            cypos = acc.y;
            czpos = acc.z;

#ifdef ALREADYTRAINED
            // normalize the IMU data between 0 to 1 and store in the model's
            // input tensor
            TFModel::tflInputTensor->data.f[samplesRead * 3 + 0] = (cxpos + 2048.0) / 4096.0;
            TFModel::tflInputTensor->data.f[samplesRead * 3 + 1] = (cypos + 2048.0) / 4096.0;
            TFModel::tflInputTensor->data.f[samplesRead * 3 + 2] = (czpos + 2048.0) / 4096.0;
            // TFModel::tflInputTensor->data.i16[samplesRead * 3 + 0] = cxpos + 2048.0;
            // TFModel::tflInputTensor->data.i16[samplesRead * 3 + 1] = cypos + 2048.0;
            // TFModel::tflInputTensor->data.i16[samplesRead * 3 + 2] = czpos + 2048.0;
#endif
            Serial.print(TFModel::tflInputTensor->data.f[samplesRead * 3 + 0]);
            Serial.print(',');
            Serial.print(TFModel::tflInputTensor->data.f[samplesRead * 3 + 1]);
            Serial.print(',');
            Serial.print(TFModel::tflInputTensor->data.f[samplesRead * 3 + 2]);
            Serial.print(',');
            Serial.print(samplesRead * 3 + 0);
            Serial.print(',');
            Serial.print(samplesRead * 3 + 1);
            Serial.print(',');
            Serial.print(samplesRead * 3 + 2);
            Serial.println();

            samplesRead++;

            if (samplesRead == numSamples)
            {
#ifdef ALREADYTRAINED
                // Run inferencing
                TfLiteStatus invokeStatus = TFModel::tflInterpreter->Invoke();
                if (invokeStatus != kTfLiteOk)
                {
                    Serial.println("Invoke failed!");
                    while (1)
                        ;
                }
                int bestAnswer = -1;
                // Loop through the output tensor values from the model
                // for (int i = 0; i < 357; i++){
                    for (int i = 0; i < NUM_GESTURES; i++)
                    {
                        Serial.print(GESTURES[i]);
                        Serial.print(": ");
                        Serial.print(TFModel::tflOutputTensor->data.i16[outputCount * NUM_GESTURES + i]);
                        Serial.print(" ");
                        Serial.println((TFModel::tflOutputTensor->data.i16[outputCount * NUM_GESTURES + i] + 2048.0) / 4096.0);
                        Serial.print(" outputCount: ");
                        Serial.println(outputCount);
                        if (((TFModel::tflOutputTensor->data.i16[outputCount * NUM_GESTURES + i] + 2048.0) / 4096.0) > 0.95)
                        {
                            bestAnswer = i;
                            break;
                        }
                    }
                // }

                outputCount++;
                if (bestAnswer > -1)
                {
                    Serial.print("*** I Think it was a ");
                    Serial.print(GESTURES[bestAnswer]);
                    Serial.println("***");
                    // MicroDebug.sendUserMessage(GESTURES[bestAnswer]);
                    tft->fillScreen(TFT_BLACK);
                    tft->setTextColor(TFT_GREEN);
                    tft->drawString(GESTURES[bestAnswer], 25, 50, 4);
                    lastReadingDone = millis();
                }
                else
                {
                    tft->fillScreen(TFT_BLACK);
                    tft->setTextColor(TFT_GREEN);
                    tft->drawString("  ???   ", 25, 50, 4);
                    // Dont update last reading so re-assesses as fast as
                    // possible
                }
#else

                tft->fillScreen(TFT_BLACK);
                tft->setTextColor(TFT_RED, TFT_BLACK);
                tft->drawString(" !WAIT! ", 25, 50, 4);
                lastReadingDone = millis();
#endif

                // add an empty line if it's the last sample
                Serial.println();
                delay(100);
                // MicroDebug.sendUserMessage("\n");
            }
        }
    }
    // unsigned long start_timestamp = micros();
    // float x_val = 1.2;

    // TFModel::tflInputTensor->data.f[0] = 2000;
    // TFModel::tflInputTensor->data.f[1] = 1000;
    // TFModel::tflInputTensor->data.f[2] = 9;

    // TfLiteStatus invoke_status = TFModel::tflInterpreter->Invoke();
    // if (invoke_status != kTfLiteOk)
    // {
    //     TFModel::tflErrorReporter->Report("Invoke failed on input: %f\n", x_val);
    // }

    // float y_val = TFModel::tflOutputTensor->data.f[0];

    // Serial.println(y_val);

    // Serial.print("Time for inference (us): ");
    // Serial.println(micros() - start_timestamp);

    // delay(5000);
}