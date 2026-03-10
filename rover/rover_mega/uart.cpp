/**
 * uart.cpp
 * Created 9/6/2025
 * 
 * Handles the reception and execution of Raspberry Pi commands and the
 * composition and transmission of telemetry strings over Serial1.
 */

#include "uart.h"
#include "config.h"
#include "state.h"
#include "motors.h"
#include "sensors.h"

static const char* motor_names[6]      = {"LF","LM","LR","RF","RM","RR"};
static const char* ultrasonic_names[5] = {"USLI","USLF","USCT","USRT","USRR"};

/*
Captures, processes, and executes byte command from Raspberry Pi.
Command structure: 
  bit 7: 0 for MOVE, 1 for TURN
  bit 6: [0 for FWD, 1 for REV] (MOVE), [0 for LEFT, 1 for RIGHT] (TURN)
  bits 0-5: speed (0 to 220.4 rpm with a resolution of about 3.5 rpm)
*/
void uart_do_command() 
{
  last_command_time_us = micros();

  if (Serial1.available() < 1) return;
  const int cmd_byte = Serial1.read();
  const uint8_t cmd = uint8_t(cmd_byte != -1 ? cmd_byte : 0);
  const uint8_t opcode = cmd >> 7;        // 1 for TURN, 0 for MOVE

  switch (opcode) 
  {
    case 0: // MOVE
    { 
      const move_dir dir = ((cmd >> 6) & 0x1)
                           ? move_dir::reverse 
                           : move_dir::forward;
      const int16_t spd = int16_t(cmd & 0x3F) * (dir == move_dir::forward ? 1 : -1);
      const int16_t rpm = map(spd, -0x3F, 0x3F, -max_rpm, max_rpm);

      // Verify that it is safe to move (Always safe to reverse)
      bool ok = true;//!ultrasonics_attached || dir == move_dir::reverse;
      if (!ok) 
      {
        // get latest distances (with tear protection)
        float d0, d1, d2;
        noInterrupts();
        d0 = ultrasonic_cm[0]; d1 = ultrasonic_cm[1]; d2 = ultrasonic_cm[2];
        interrupts();

        //ok = (d0 > safe_dist_cm && d1 > safe_dist_cm && d2 > safe_dist_cm);
      }
      if (ok) 
      {
        //Serial.print("CMD MOVE: Dir="); Serial.print((dir == move_dir::forward) ? "FWD" : "REV");
        //Serial.print(" RPM="); Serial.println(rpm);
        motors_move(dir, rpm);
        last_move_time_us = micros();
        if (rpm != 0) ugv_is_moving = true;
      }
      break;
    }

    case 1: // TURN
    {
      const move_dir spin = ((cmd >> 6) & 0x1)
                            ? move_dir::right_spin 
                            : move_dir::left_spin;
      const uint8_t rpm = map((cmd & 0x3F)*4, min_pw, max_pw, min_rpm, max_rpm);

      motors_move(spin, rpm);
      last_move_time_us = micros();
      if (rpm != 0) ugv_is_moving = true;
      
      break;
    }

    default: break;
  }

}

/*
Sends a telemetry string over Serial1 to the Raspberry Pi.

Telemetry format (on one line, breaks here for readability):
TIME=seconds|
LFV=float|LFA=float|LFR=int|LMV=float|LMA=float|LMR=int|
LRV=float|LRA=float|LRR=int|RFV=float|RFA=float|RFR=int|
RMV=float|RMA=float|RMR=int|RRV=float|RRA=float|RRR=int|
USLI=float|USLF=float|USCT=float|USRT=float|USRR=float|
R=float|P=float|Y=float|AX=float|AY=float|AZ=float|
TEMP=float|RHUM=float|LVIS=int|LINF=int|
*/
void uart_send_telemetry() 
{
  // Get and reset per-second averages
  float rpm_avg[6], mot_v_avg[6], mot_a_avg[6];
  if (motors_attached) 
  {
    motors_get_and_reset_rpm_avg(rpm_avg);
    motors_get_and_reset_pow_avg(mot_v_avg, mot_a_avg);
  }
  sensor_avgs env{};
  if (env_sensors_attached) sensors_get_and_reset_env_avg(env);

  imu_avgs imu_avg{};
  if (imu_attached) sensors_get_and_reset_imu_avg(imu_avg);
  
  float us_avg[5];
  if (ultrasonics_attached) sensors_get_and_reset_ultra_avg(us_avg);

  float batt_v_avg, batt_a_avg, batt_pct_avg;
  sensors_get_and_reset_batt_avg(batt_v_avg, batt_a_avg, batt_pct_avg);


  // Build telemetry string
  String t_str; 
  t_str.reserve(256);
  t_str  = "TIME=" + String(float(millis())/1000.0f, 3) + "|";

  if (motors_attached) 
  {
    for (int i = 0; i < 6; i++) 
    {
      t_str += motor_names[i];  t_str += "V=" + String(mot_v_avg[i], 4) + "|";
      t_str += motor_names[i];  t_str += "A=" + String(mot_a_avg[i], 4) + "|";
      t_str += motor_names[i];  t_str += "R="; 
      int rpm_i = int(rpm_avg[i] + (rpm_avg[i] >= 0 ? 0.5f : -0.5f)); // round
      t_str += String(rpm_i);
      t_str += "|";
    }
  }
  else 
  {
    t_str += "LFV=0|LFA=0|LFR=0|LMV=0|LMA=0|LMR=0|LRV=0|LRA=0|LRR=0|"
             "RFV=0|RFA=0|RFR=0|RMV=0|RMA=0|RMR=0|RRV=0|RRA=0|RRR=0|";
  }

  if (ultrasonics_attached) 
  {
    for (int i = 0; i < num_ultrasonics; i++)
      t_str += String(ultrasonic_names[i]) + "=" + String(us_avg[i], 1) + "|";
  } 
  else { t_str += "USLI=0|USLF=0|USCT=0|USRT=0|USRR=0|"; }

  if (imu_attached)
  {
    t_str += "R=" + String(imu_avg.pose.roll, 1) + "|";
    t_str += "P=" + String(imu_avg.pose.pitch, 1) + "|";
    t_str += "Y=" + String(imu_avg.pose.yaw, 1) + "|";
    t_str += "AX=" + String(imu_avg.accx, 4) + "|";
    t_str += "AY=" + String(imu_avg.accy, 4) + "|";
    t_str += "AZ=" + String(imu_avg.accz, 4) + "|";
    
  }
  else { t_str += "R=0|P=0|Y=0|AX=0|AY=0|AZ=0|"; }

  if (env_sensors_attached) 
  {
    t_str += "TEMP=" + String(env.temp_c, 1)  + "|";
    t_str += "RHUM=" + String(env.rel_hum, 2) + "|";
    t_str += "LVIS=" + String(env.visible)    + "|";
    t_str += "LINF=" + String(env.infrared)   + "|";
  }
  else { t_str += "TEMP=0|RHUM=0|LVIS=0|LINF=0|"; }

  t_str += "BV=" + String(batt_v_avg, 2)     + "|";
  t_str += "BA=" + String(batt_a_avg, 2)     + "|";
  t_str += "BPCT=" + String(batt_pct_avg, 1) + "|";

  //Serial.println(t_str);       // Displays telemetry string over USB
  Serial1.println(t_str);        // Sends telemetry string to Raspberry Pi
  last_talk_time_us = micros();
}
