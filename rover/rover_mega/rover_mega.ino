/**
 * rover_mega.ino
 * Created 9/6/2025
 * 
 * The Arduino equivalent of main.cpp. Contains setup() and loop() functions.
 * Calls functions to configure attached subsystems and opens serial ports. 
 * Once they are open, enters main loop which periodically samples sensors and
 * encoders, performs commands, and transmits telemetry once per second.
 * 
 * NOTE: To flash the Mega using the Raspberry Pi command line:
 *    0. Install arduino-cli (see docs), plug in Arduino, go to the repo root
 *    1. arduino-cli compile --fqbn arduino:avr:mega ./rover/rover_mega
 *    2. arduino-cli upload -p /dev/ttyACM0 --fqbn arduino:avr:mega ./rover/rover_mega -v
 *    3. arduino-cli monitor -p /dev/ttyACM0 -b arduino:avr:mega -c 460800
 * NOTE: Step 3 will only work if you uncomment the serial print in uart.cpp. 
 *    Make sure that the baudrate matches the baudrate in config.h.
 * NOTE: Serial prints are slow as hell. Don't use them if you can avoid it. 
 *    They will slow down the entire telemetry system severely. If you do, 
 *    stick to the highest baud you can. I had success with 460,800 bps.
 */

#include <Arduino.h>

#include "config.h"
#include "interrupts.h"
#include "state.h"
#include "uart.h"
#include "motors.h"
#include "sensors.h"

void setup() 
{
  // Set up PWM timers for 31 kHz operation to avoid audible motor whine
  // Timer 3 (OC3B=PE4=D2=RR, OC3C=PE5=D3=RF, +D5)
  TCCR3B &= ~7; // Flush bits 0-2 (timer prescale)
  TCCR3B |=  1; // Prescale = 1, PWM F=31 kHz

  // Timer 4 (OC4A=PH3=D6=LR, OC4B=PH4=D7=LF, +D8)
  TCCR4B &= ~7; // Flush bits 0-2 (timer prescale)
  TCCR4B |=  1; // Prescale = 1, PWM F=31 kHz

  if (motors_attached)                                { motors_setup(); }
  if (ultrasonics_attached 
      || env_sensors_attached
      || imu_attached)                                { sensors_setup(); }
  if (encoders_attached || ultrasonics_attached)      { interrupts_setup(); }

  if (uart_attached) 
  {
    Serial.begin(mega_baudrate);
    Serial1.begin(ugv_baudrate);
    while (!Serial1) {}   // Block until backplane connection is ready
  }
}

void loop() 
{
  const uint64_t now_us = micros();

  // Periodic sensor and encoder sampling
  if (ultrasonics_attached 
    && (now_us - last_ultra_sample_us) >= ultrasonic_sample_period_us) 
  {
    sensors_ultrasonics_tick(now_us);
    // updates last_ultra_sample_us internally
  }

  if (imu_attached
    && (now_us - last_imu_sample_us) >= imu_sample_period_us)
  {
    sensors_imu_tick(now_us);
    last_imu_sample_us = now_us;
  }

  if (env_sensors_attached 
    && (now_us - last_env_sample_us) >= sensor_sample_period_us) 
  {
    sensors_env_tick(now_us);
    last_env_sample_us = now_us;
  }

  if (encoders_attached 
    && (now_us - last_encoder_sample_us) >= encoder_sample_period_us) 
  {
    motors_encoder_tick();
    last_encoder_sample_us = now_us;
  }

  if (motors_attached
    && (now_us - last_power_sample_us) >= power_sample_period_us)
  {
    motors_power_tick();
    last_power_sample_us = now_us;
  }

  if (now_us - last_power_sample_us >= power_sample_period_us) 
  {
    sensors_batt_tick(now_us);
    last_power_sample_us = now_us;
  }

  // Movement command processing and execution
  if (uart_attached) 
  {
    if (now_us - last_command_time_us > command_threshold_us) 
    { 
      uart_do_command();
    } 
    // This prevents motor jerk due to brief gap between commands
    // 3 could maybe be 2.5 for lower command latency
    else if (ugv_is_moving 
          && (now_us - last_move_time_us > (3 * command_threshold_us))) 
    {
      motors_stop();
      ugv_is_moving = false;
      last_move_time_us = now_us;
    }
  }

  // Telemetry generation and transmission
  if (uart_attached 
    && (now_us - last_talk_time_us) >= telemetry_period_us) 
  {
    if (skip_first_telemetry) 
    { 
        skip_first_telemetry = false;
        last_talk_time_us = now_us; 
    }
    else { uart_send_telemetry(); } // updates last_talk_time_us internally
  }
}
