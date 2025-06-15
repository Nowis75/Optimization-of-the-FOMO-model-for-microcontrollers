#include "Arduino.h"
#include "esp_camera.h"

#include "fomo_model.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"

#define CAMERA_MODEL_TSIMCAM_ESP32S3
#include "camera_pins.h"

#define IMG_WIDTH 96
#define IMG_HEIGHT 96
#define GRID_SIZE 12
#define INPUT_LEN (IMG_WIDTH * IMG_HEIGHT)
#define OUTPUT_LEN (GRID_SIZE * GRID_SIZE)
#define THRESHOLD 0.5f
#define TENSOR_ARENA_SIZE (350 * 1024)

uint8_t* tensor_arena = nullptr;

const tflite::Model* model;
tflite::MicroInterpreter* interpreter;
tflite::ErrorReporter* error_reporter;
TfLiteTensor* input_tensor;
TfLiteTensor* output_tensor;

static camera_config_t camera_config = {
    .pin_pwdn = PWDN_GPIO_NUM,
    .pin_reset = RESET_GPIO_NUM,
    .pin_xclk = XCLK_GPIO_NUM,
    .pin_sscb_sda = SIOD_GPIO_NUM,
    .pin_sscb_scl = SIOC_GPIO_NUM,
    .pin_d7 = Y9_GPIO_NUM,
    .pin_d6 = Y8_GPIO_NUM,
    .pin_d5 = Y7_GPIO_NUM,
    .pin_d4 = Y6_GPIO_NUM,
    .pin_d3 = Y5_GPIO_NUM,
    .pin_d2 = Y4_GPIO_NUM,
    .pin_d1 = Y3_GPIO_NUM,
    .pin_d0 = Y2_GPIO_NUM,
    .pin_vsync = VSYNC_GPIO_NUM,
    .pin_href = HREF_GPIO_NUM,
    .pin_pclk = PCLK_GPIO_NUM,
    .xclk_freq_hz = 20000000,
    .ledc_timer = LEDC_TIMER_0,
    .ledc_channel = LEDC_CHANNEL_0,
    .pixel_format = PIXFORMAT_GRAYSCALE,
    .frame_size = FRAMESIZE_QQVGA,
    .jpeg_quality = 12,
    .fb_count = 1,
    .fb_location = CAMERA_FB_IN_PSRAM,
    .grab_mode = CAMERA_GRAB_WHEN_EMPTY
};

bool initCamera() {
    if (esp_camera_init(&camera_config) != ESP_OK) {
        Serial.println("Camera init failed");
        return false;
    }
    return true;
}

void setup() {
    Serial.begin(115200);
    delay(500);
    Serial.println("FOMO Detector using TFLite Micro");

    if (!initCamera()) {
        while (true);
    }

    static tflite::MicroErrorReporter micro_error_reporter;
    error_reporter = &micro_error_reporter;

    model = tflite::GetModel(fomo_detector_int8_tflite);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        Serial.println("Model schema version mismatch");
        while (true);
    }

    tensor_arena = (uint8_t*) ps_malloc(TENSOR_ARENA_SIZE);
    if (!tensor_arena) {
        Serial.println("Failed to allocate tensor_arena in PSRAM");
        while (true);
    }

    static tflite::AllOpsResolver resolver;
    static tflite::MicroInterpreter static_interpreter(
        model, resolver, tensor_arena, TENSOR_ARENA_SIZE, error_reporter);
    interpreter = &static_interpreter;

    if (interpreter->AllocateTensors() != kTfLiteOk) {
        Serial.println("Failed to allocate tensors");
        while (true);
    }

    input_tensor = interpreter->input(0);
    output_tensor = interpreter->output(0);

    Serial.println("Model initialized");
}

void loop() {
    camera_fb_t* frame = esp_camera_fb_get();
    if (!frame) {
        Serial.println("Failed to capture frame");
        return;
    }

    int idx = 0;
    for (int y = 0; y < IMG_HEIGHT; y++) {
        int src_y = y * frame->height / IMG_HEIGHT;
        for (int x = 0; x < IMG_WIDTH; x++) {
            int src_x = x * frame->width / IMG_WIDTH;
            int pixel = frame->buf[src_y * frame->width + src_x];
            input_tensor->data.int8[idx++] = pixel - 128;
        }
    }
    esp_camera_fb_return(frame);

    if (interpreter->Invoke() != kTfLiteOk) {
        Serial.println("Inference failed");
        return;
    }

    Serial.println("Detection grid:");
    char grid[GRID_SIZE * (GRID_SIZE + 1) + 1]; // včetně '\n' a '\0'
    int ptr = 0;

    for (int y = 0; y < GRID_SIZE; y++) {
        for (int x = 0; x < GRID_SIZE; x++) {
            int i = y * GRID_SIZE + x;
            float value = (output_tensor->data.int8[i] + 128) * 0.00390625f;
            grid[ptr++] = value > THRESHOLD ? 'x' : '.';
        }
        grid[ptr++] = '\n';
    }
    grid[ptr] = '\0';
    Serial.print(grid);

}
