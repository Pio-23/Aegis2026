/**
 * sensors.h
 * Created 9/6/2025
 * 
 * Provides functions to sample ultrasonic, light, and temp sensors, 
 * and offload sampling calculations from ISRs. 
 */

#ifndef AEGIS_SENSORS_H
#define AEGIS_SENSORS_H
#include <Arduino.h>
#include "config.h"

// Environmental (SHTC3 + LTR-329) sample data
struct sensor_avgs {
  float    temp_c = 0.0f;
  float    rel_hum = 0.0f;
  uint16_t visible = 0;
  uint16_t infrared = 0;
};

struct imu_pose_quat {
  float r = 0.0f;
  float i = 0.0f;
  float j = 0.0f;
  float k = 0.0f;
};

struct imu_pose_euler {
  float roll  = 0.0f;
  float pitch = 0.0f;
  float yaw   = 0.0f;
};

struct imu_avgs {
  imu_pose_euler pose;

  float accx  = 0.0f;
  float accy  = 0.0f;
  float accz  = 0.0f;
};

void sensors_setup();

void sensors_env_tick(uint32_t now_us);
void sensors_imu_tick(uint32_t now_us);
void sensors_ultrasonics_tick(uint32_t now_us);
void sensors_batt_tick(uint32_t now_us);

void sensors_get_and_reset_env_avg(sensor_avgs& out);
void sensors_get_and_reset_imu_avg(imu_avgs& out);
void sensors_get_and_reset_ultra_avg(float out_cm[num_ultrasonics]);
void sensors_get_and_reset_batt_avg(float& out_v, float& out_a, float& out_pct);

void sensors_handle_pcint0_echoes();

void get_euler_from_quaternion(imu_avgs& out, imu_pose_quat q);

void control_headlights(uint16_t vis_lux);

#endif
