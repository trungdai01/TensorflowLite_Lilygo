

#include "../include/global.h"
#include "TensorFlowLite_ESP32.h"
#include "sine_model.h"
// #include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

constexpr float pi = 3.14159265;
constexpr float freq = 100;
constexpr float period = (1 / freq) * (1000000);

namespace TFModel
{
    tflite::ErrorReporter *error_reporter = NULL;
    const tflite::Model *model = NULL;
    tflite::MicroInterpreter *interpreter = NULL;
    TfLiteTensor *model_input = NULL;
    TfLiteTensor *model_output = NULL;

    constexpr int kTensorArenaSize = 8 * 1024;
    uint8_t tensor_arena[kTensorArenaSize];
}  // namespace TFModel

void setup()
{
    Serial.begin(115200);
    while (!Serial)
        ;
    static tflite::MicroErrorReporter micro_error_reporter;
    TFModel::error_reporter = &micro_error_reporter;
    TFModel::model = tflite::GetModel(sine_model);
    if (TFModel::model->version() != TFLITE_SCHEMA_VERSION)
    {
        TFModel::error_reporter->Report("Model Version does not match SCHEMA");
        while (1)
            ;
    }

    static tflite::MicroMutableOpResolver<1> micro_mutable_op_resolver;
    micro_mutable_op_resolver.AddFullyConnected();

    static tflite::MicroInterpreter static_interpreter(TFModel::model, micro_mutable_op_resolver, TFModel::tensor_arena, TFModel::kTensorArenaSize, TFModel::error_reporter);
    TFModel::interpreter = &static_interpreter;
    TfLiteStatus allocate_status = TFModel::interpreter->AllocateTensors();
    if (allocate_status != kTfLiteOk)
    {
        TFModel::error_reporter->Report("AllocateTensors() failed!");
        while (1)
            ;
    }
    TFModel::model_input = TFModel::interpreter->input(0);
    TFModel::model_output = TFModel::interpreter->output(0);

    Serial.print("Number of dimensions: ");
    Serial.println(TFModel::model_input->dims->size);
    Serial.print("Dim 1 size: ");
    Serial.println(TFModel::model_input->dims->data[0]);
    Serial.print("Dim 2 size: ");
    Serial.println(TFModel::model_input->dims->data[1]);
    Serial.print("Input type: ");
    Serial.println(TFModel::model_input->type);
}

void loop()
{
    unsigned long start_timestamp = micros();
    float x_val = 1.2;

    TFModel::model_input->data.f[0] = x_val;
    TFModel::model_input->data.f[1] = 1.3;
    TFModel::model_input->data.f[2] = 1.4;
    TFModel::model_input->data.f[3] = 0.9;

    TfLiteStatus invoke_status = TFModel::interpreter->Invoke();
    if(invoke_status != kTfLiteOk)
    {
        TFModel::error_reporter->Report("Invoke failed on input: %f\n", x_val);
    }

    float y_val = TFModel::model_output->data.f[0];

    Serial.println(TFModel::model_output->data.f[0]);
    Serial.println(TFModel::model_output->data.f[1]);
    Serial.println(TFModel::model_output->data.f[2]);
    Serial.println(TFModel::model_output->data.f[3]);

    Serial.print("Time for inference (us): ");
    Serial.println(micros() - start_timestamp);

    delay(5000);
}