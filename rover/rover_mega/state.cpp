/**
 * state.cpp
 * Created 9/6/2025
 * 
 * Initializes and stores state variables, e.g. timestamps and ISR-shared sets.
 */

#include "state.h"

// Initialize times to zero
uint32_t last_command_time_us   = 0;
uint32_t last_move_time_us      = 0;
uint32_t last_ultra_sample_us   = 0;
uint32_t last_env_sample_us     = 0;
uint32_t last_imu_sample_us     = 0;
uint32_t last_encoder_sample_us = 0;
uint32_t last_power_sample_us   = 0;
uint32_t last_talk_time_us      = 0;
uint32_t last_batt_sample_us    = 0;

bool ugv_is_moving       = false;
bool skip_first_telemetry = true;

volatile int16_t enc_pulse_counts[6] = {0,0,0,0,0,0};
volatile float ultrasonic_cm[num_ultrasonics] = {0,0,0};
