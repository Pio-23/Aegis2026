/**
 * motors.h
 * Created 9/6/2025
 * 
 * Provides movement patterns and functions that generate PWM signals, sample 
 * and reset encoders, and offload sampling calculations from ISRs.
 */

#ifndef AEGIS_MOTORS_H
#define AEGIS_MOTORS_H
#include <Arduino.h>

enum class move_dir : uint8_t {stop, forward, reverse, left_spin, right_spin};

void motors_setup();
void motors_move(move_dir dir, int16_t rpm);
void motors_stop();

void motors_encoder_tick();
void motors_power_tick();

move_dir motors_calculate_pid_rpms(int16_t target);

void motors_get_and_reset_rpm_avg(float out_avg_rpm[6]);
void motors_get_and_reset_pow_avg(float out_avg_v[6], float out_avg_a[6]);

void motors_handle_pcint0_encoders();
void motors_handle_pcint1_encoders();
#endif
