/**
 * config.h
 * Created 9/6/2025
 * 
 * Holds all pins, parameters, and flags used by other source files.
 * 
 * Temperature, ambient light, and IMU sensors are on I2C SCL and SDA pins
 * but are handled internally by arduino libraries and are not listed here.
 */

#ifndef AEGIS_CONFIG_H
#define AEGIS_CONFIG_H
#include <Arduino.h>

// Subsytem Flags
// ENV SENSORS BLOCK IF MARKED TRUE AND DISCONNECTED, FIX
constexpr bool uart_attached        = true;  // ARD <-> RPI
constexpr bool motors_attached      = true;  // Yellowjackets
constexpr bool encoders_attached    = true;  // On motors
constexpr bool env_sensors_attached = true;  // SHTC3 + LTR-329 
constexpr bool imu_attached         = true;  // BNO-085
constexpr bool ultrasonics_attached = true;  // HC-SR04
constexpr bool headlights_attached  = true;  // KC LED Headlights

// Serial Parameters
constexpr uint32_t mega_baudrate = 460800;     // Baud of serial
constexpr uint32_t ugv_baudrate  = 115200;     // Baud of serial1

// Timing Parameters
constexpr uint32_t telemetry_period_us         = 1000000;  //  1 Hz
constexpr uint32_t ultrasonic_sample_period_us =  100000;  // 10 Hz
constexpr uint32_t sensor_sample_period_us     =  100000;  // 10 Hz
constexpr uint32_t imu_sample_period_us        =  100000;  // 10 Hz
constexpr uint32_t encoder_sample_period_us    =   99000;  // 10.1 Hz
constexpr uint32_t power_sample_period_us      =  100000;  // 10 Hz
constexpr uint32_t command_threshold_us        =   50000;  // 20 Hz

// Encoder Characteristics
constexpr float enc_pulses_per_rev = 753.2 / 4; // Four "events", see datasheet
constexpr float enc_gear_ratio     = 26.9;      // Encoder vs. motor shaft

// Motor Constraints
constexpr uint8_t min_rpm = 0;
constexpr uint8_t max_rpm = 100;
constexpr uint8_t min_pw  = 50;
constexpr uint8_t max_pw  = 255;

// Motor Pins
constexpr uint8_t left_fwd  = 6;
constexpr uint8_t left_rvs  = 7;
constexpr uint8_t right_fwd = 3;
constexpr uint8_t right_rvs = 2;
constexpr uint8_t driver_pins[4] = {left_fwd, left_rvs, right_fwd, right_rvs};

// Encoder Pins                    LF, LM, LR, RF, RM, RR
constexpr uint8_t enc_a_pins[6] = {13, 12, 11, 10, 15, 14};
constexpr uint8_t enc_b_pins[6] = {33, 35, 37, 39, 41, 43};

// Ultrasonic Pins, Parameters
constexpr uint8_t  num_ultrasonics = 5;             //  1,  2,  3,  4,  5
                                                    // LI, LF, CT, RT, RR
constexpr uint8_t  ultra_trig_pins[num_ultrasonics] = {48, 46, 42, 44, 49};
constexpr uint8_t  ultra_echo_pins[num_ultrasonics] = {51, 53, 50, 52, A15};

constexpr uint32_t trig_pulse_us = 10;
constexpr int      safe_dist_cm = -1;            // Disabled for testing
constexpr float    speed_of_sound_mps = 345.0;   // 343 @ 20C + 0.6 per degC

// Headlight Pins, Parameters
constexpr uint8_t hl_highbeam_pin = 22;
constexpr uint8_t hl_left_pin = 24;
constexpr uint8_t hl_right_pin = 26;
constexpr uint16_t threshold_ambient_light = 100; // Lumens

// Power Meter Pins
constexpr uint8_t volt_lf = A11;
constexpr uint8_t volt_lm = A12;
constexpr uint8_t volt_lr = A13;
constexpr uint8_t volt_rf = A8;
constexpr uint8_t volt_rm = A9;
constexpr uint8_t volt_rr = A10;
constexpr uint8_t volt_batt = A14;
constexpr uint8_t mot_v_pins[6] = {
    volt_lf, volt_lm, volt_lr, volt_rf, volt_rm, volt_rr
};

constexpr uint8_t amp_lf = A3;
constexpr uint8_t amp_lm = A4;
constexpr uint8_t amp_lr = A5;
constexpr uint8_t amp_rf = A0;
constexpr uint8_t amp_rm = A1;
constexpr uint8_t amp_rr = A2;
constexpr uint8_t mot_a_pins[6] = {
    amp_lf, amp_lm, amp_lr, amp_rf, amp_rm, amp_rr
};
constexpr uint8_t amp_batt = A6;

#endif
