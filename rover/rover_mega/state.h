/**
 * state.h
 * Created 9/6/2025
 * 
 * Initializes and stores state variables, e.g. timestamps and ISR-shared sets.
 */

#ifndef AEGIS_STATE_H
#define AEGIS_STATE_H
#include <Arduino.h>
#include "config.h"

// Timestamps (updated from main thread)
extern uint32_t last_command_time_us;
extern uint32_t last_move_time_us;
extern uint32_t last_ultra_sample_us;
extern uint32_t last_env_sample_us;
extern uint32_t last_imu_sample_us;
extern uint32_t last_encoder_sample_us;
extern uint32_t last_power_sample_us;
extern uint32_t last_talk_time_us;
extern uint32_t last_batt_sample_us;

extern bool ugv_is_moving;
extern bool skip_first_telemetry;

// ISR-shared sets
extern volatile int16_t enc_pulse_counts[6];
extern volatile uint8_t  enc_directions[6];
extern volatile float ultrasonic_cm[num_ultrasonics];

#endif
