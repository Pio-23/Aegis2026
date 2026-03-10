/**
 * motors.cpp
 * Created 9/6/2025
 * 
 * Provides movement patterns and functions that generate PWM signals, sample 
 * and reset encoders, and offload sampling calculations from ISRs.
 */

#include <avr/interrupt.h>
#include "motors.h"
#include "config.h"
#include "state.h"

// Movement patterns                    LF, LR, RF, RR
static uint8_t stop_pattern[4]       = { 0,  0,  0,  0};
static uint8_t fwd_pattern[4]        = { 1,  0,  1,  0};
static uint8_t rev_pattern[4]        = { 0,  1,  0,  1};
static uint8_t left_spin_pattern[4]  = { 0,  1,  1,  0};
static uint8_t right_spin_pattern[4] = { 1,  0,  0,  1};

static float mot_v_inst[6] = {0,0,0,0,0,0}; // Latest instantaneous voltage
static float mot_v_sum[6] = {0,0,0,0,0,0};  // Motor voltage accumulators
static float mot_a_inst[6] = {0,0,0,0,0,0}; // Latest instantaneous current
static float mot_a_sum[6] = {0,0,0,0,0,0};  // Motor current accumulators
static uint16_t mot_pow_count = 0;          // N=0 averages are skipped

static float rpm_inst[6] = {0,0,0,0,0,0};   // Latest instantaneous rpm
static float rpm_sum[6] = {0,0,0,0,0,0};    // Encoder rpm accumulators
static int16_t rpm_count = 0;              // N=0 averages are skipped

static float rpm_prev[6] = {0,0,0,0,0,0};
static float rpm_pid[6] = {0,0,0,0,0,0};   // target rpms after PID control
static float avg_rpm_pid[2] = {0, 0}; // left, right

/*
Sets PWM on given pin to correspond with given RPM value.
*/
static inline void set_rpm_pwm(uint8_t pin, uint8_t rpm)
{
  if (rpm == 0) analogWrite(pin, 0);
  else analogWrite(pin, map(rpm, min_rpm, max_rpm, min_pw, max_pw));
}

/*
Configures motor PWM as outputs initialized to 0 and encoder pins as 
pulled-up inputs.
*/
void motors_setup()
{
  // PWM pins
  for (int i = 0; i < 4; i++)
  {
    pinMode(driver_pins[i], OUTPUT);
    analogWrite(driver_pins[i], 0);
  }

  // Encoder pins
  for (int i=0;i<6;i++) {
    pinMode(enc_a_pins[i], INPUT_PULLUP);
    pinMode(enc_b_pins[i], INPUT_PULLUP);
  }

  // Power Meter pins
  for (int i = 0; i < 6; i++)
  {
    pinMode(mot_v_pins[i], INPUT);
    pinMode(mot_a_pins[i], INPUT);
  }

}


/*
Sets all 4 motor PWM signals in accordance with matching movement pattern.
Defaults to 'stop' on no pattern match, which should never happen.
*/
void motors_move(move_dir dir, int16_t rpm)
{
  uint8_t* pattern = stop_pattern;

  // PID control if encoders are attached and moving forward/reverse
  // Flipped attached flag to disable PID hackily
  if (encoders_attached && (dir == move_dir::forward || dir == move_dir::reverse))
  {
    // Serial.print("Time: ");
    // Serial.print(millis());
    // Serial.print(" T: "); Serial.print(rpm);
    // Serial.print(" A: "); Serial.print(rpm_inst[0]); Serial.print(" "); Serial.print(rpm_inst[3]);

    // Get adjusted rpms from PID controller, round, and set PWM
    move_dir pid_pattern_enum = motors_calculate_pid_rpms(rpm);
    const int16_t adjusted_rpm[2] = {
      int16_t(avg_rpm_pid[0] + (avg_rpm_pid[0] > 0 ? 0.5 : -0.5)),
      int16_t(avg_rpm_pid[1] + (avg_rpm_pid[1] > 0 ? 0.5 : -0.5))
    };

    switch (pid_pattern_enum)
    {
      case move_dir::stop:       pattern = stop_pattern;       break;
      case move_dir::forward:    pattern = fwd_pattern;        break;
      case move_dir::reverse:    pattern = rev_pattern;        break;
      case move_dir::left_spin:  pattern = left_spin_pattern;  break;
      case move_dir::right_spin: pattern = right_spin_pattern; break;
      default:                   pattern = stop_pattern;       break;
    }

    for (int i = 0; i < 4; i++) {
      set_rpm_pwm(driver_pins[i], abs(adjusted_rpm[i/2]) * *(pattern + i));
    }
  }
  else
  {

    switch (dir)
    {
      case move_dir::stop:       pattern = stop_pattern;       break;
      case move_dir::forward:    pattern = fwd_pattern;        break;
      case move_dir::reverse:    pattern = rev_pattern;        break;
      case move_dir::left_spin:  pattern = left_spin_pattern;  break;
      case move_dir::right_spin: pattern = right_spin_pattern; break;
      default:                   pattern = stop_pattern;       break;
    }

    // Without encoders, no PID control; set all PWMs directly
    for (int i = 0; i < 4; i++) {
      set_rpm_pwm(driver_pins[i], abs(rpm) * *(pattern + i));
    }
  }
  
}


void motors_stop() { motors_move(move_dir::stop, 0); }


/*
Calculates target RPMs for left and right motors using PID control based on
the given target RPM and the last measured instantaneous RPMs.
*/
move_dir motors_calculate_pid_rpms(int16_t target) {
  constexpr float kp = 0.48f;
  constexpr float ki = 1.92f; // Dropped from 1.92f to reduce reverse overshoot
  constexpr float kd = 0.036f;
  constexpr float dt = encoder_sample_period_us * 1e-6;

  static float integrals[6] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  static float diffs[6]     = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

  avg_rpm_pid[0] = 0; avg_rpm_pid[1] = 0;

  static int16_t last_target = 0;
  if ((target > 0 && last_target < 0) ||
      (target < 0 && last_target > 0)) {
      for (int i = 0; i < 6; ++i) integrals[i] = 0.0f;
  }
  last_target = target;

  for (int i = 0; i < 6; i++)
  {
    float err = float(target) - rpm_inst[i];

    integrals[i] += err * dt;
    if (integrals[i] > max_rpm)        integrals[i] = max_rpm;
    else if (integrals[i] < -max_rpm)  integrals[i] = -max_rpm;

    diffs[i] = -1 * (rpm_inst[i] - rpm_prev[i]) / dt;

    rpm_pid[i] = rpm_inst[i] + kp * err + ki * integrals[i] + kd * diffs[i];
    
    if (rpm_pid[i] > max_rpm)        rpm_pid[i] = max_rpm;
    else if (rpm_pid[i] < -max_rpm)  rpm_pid[i] = -max_rpm;

    // Switched from average left and right to front left and right for PID tracking
    // If this works, refactor PID to not waste time on the other four motors
    // Pick one from 0-2 and one from 3-5.
    if (i == 0 || i == 5)
    {
      avg_rpm_pid[i/3] += rpm_pid[i];
    }
  }

  if (     (avg_rpm_pid[0] < 0) && (avg_rpm_pid[1] < 0)) return move_dir::reverse;
  else if ((avg_rpm_pid[0] > 0) && (avg_rpm_pid[1] > 0)) return move_dir::forward;
  else return move_dir::stop;

}


/*
Takes a reading of (and clears) motor encoder pulse counts, converts them into
instantaneous RPM measurements, and appends them to their accumulators. This
function is called every N microseconds, where N is the period from config.h.
*/
void motors_encoder_tick()
{
  // Capture and clear encoder pulses (disable interrupts to avoid tearing)
  int16_t counts[6];
  noInterrupts();
  for (int i = 0; i < 6; i++) 
  {
    counts[i] = enc_pulse_counts[i];
    enc_pulse_counts[i] = 0;
  }
  interrupts();

  // Calculate, record, and accumulate instantaneous RPMs
  const float window_s = 0.099f;
  for (int i = 0; i < 6; i++)
  {
    rpm_prev[i] = rpm_inst[i];

    const float inst = float(counts[i]) 
                      / enc_pulses_per_rev 
                      / (encoder_sample_period_us * 1e-6)
                      * 60;
    rpm_inst[i] = inst;
    rpm_sum[i] += inst;
  }

  rpm_count++;
}


/*
Takes a reading of (and clears) motor voltages and currents, and appends them to
their accumulators. This function is called every N microseconds, where N is
the period from config.h.
*/
void motors_power_tick()
{
  static constexpr float v_ref_mot = 6.0f;     // ADC reference voltage
  static constexpr float shunt_res = 0.028f;  // Ohms
  static constexpr float v_cap_off = 0.15f;    // Volts
  static constexpr float ammeter_gain = 24.489f;
  static constexpr float v_voltage_div = 1.25f;

  // Read and accumulate motor voltages and currents
  for (int i = 0; i < 6; i++)
  {
    const float v_resolution = v_ref_mot / 1023.0f;
    const float v_inst = analogRead(mot_v_pins[i]) 
                          * v_resolution 
                          * v_voltage_div;
    const float a_inst = (analogRead(mot_a_pins[i]) * v_resolution - v_cap_off) 
                          / ammeter_gain 
                          / shunt_res;

    mot_v_inst[i] = v_inst;
    mot_a_inst[i] = (a_inst < 0.0f) ? a_inst : 0.0f; // No reverse current

    mot_v_sum[i] += v_inst;
    mot_a_sum[i] += a_inst;
  }

  mot_pow_count++;
}


/*
Takes an average of the last N RPM readings and resets the accumulators.
Average RPMs are then sent to Raspberry Pi with the rest of the telemetry.
*/
void motors_get_and_reset_rpm_avg(float out_avg_rpm[6])
{
  for (int i = 0; i < 6; i++) 
  {
    out_avg_rpm[i] = rpm_count ? (rpm_sum[i] / rpm_count) : 0.0f;
    rpm_sum[i] = 0.0f;
  }
  rpm_count = 0;
}

/*
Takes an average of the last N voltage and current readings and resets the
accumulators. Average voltages and currents are then sent to Raspberry Pi with 
the rest of the telemetry.
*/
void motors_get_and_reset_pow_avg(float out_avg_v[6], float out_avg_a[6])
{
  for (int i = 0; i < 6; i++) 
  {
    out_avg_v[i] = mot_pow_count ? (mot_v_sum[i] / mot_pow_count) : 0.0f;
    mot_v_sum[i] = 0.0f;

    out_avg_a[i] = mot_pow_count ? (mot_a_sum[i] / mot_pow_count) : 0.0f;
    mot_a_sum[i] = 0.0f;
  }
  mot_pow_count = 0;
}

/*
Handles interrupts generated by pulses from the A-channel of the motor
encoders. Adds a pulse if detected and determines direction of turn.
Four encoders are wired to pins in PCINT0 register.
*/
void motors_handle_pcint0_encoders()
{
  // Encoders on PB4-PB7
  // This probably shouldn't be hardcoded like this but oh well
  static const uint8_t pos[4]    = { 0, 1, 2, 5};
  static const uint8_t a_pins[4] = {35,37,33,15};
  static const uint8_t b_pins[4] = {12,11,13,41};
  static uint8_t a_state[4]      = { 0, 0, 0, 0};

  for (int i = 0; i < 4; i++) 
  {
    const uint8_t a = digitalRead(a_pins[i]);
    if (a && !a_state[i]) 
    {
      const uint8_t b = digitalRead(b_pins[i]);
      if (b) enc_pulse_counts[pos[i]]--;
      else   enc_pulse_counts[pos[i]]++;
      a_state[i] = 1;
    } 
    else if (!a && a_state[i]) { a_state[i] = 0; }
  }
}

/*
Handles interrupts generated by pulses from the A-channel of the motor
encoders. Adds a pulse if detected and determines direction of turn.
Two encoders are wired to pins in PCINT1 register.
*/
void motors_handle_pcint1_encoders()
{
  // Encoders on PJ0, PJ1
  // This probably shouldn't be hardcoded like this but oh well
  static const uint8_t pos[2]    = { 3, 4};
  static const uint8_t a_pins[2] = {14, 10};
  static const uint8_t b_pins[2] = {43, 39};
  static uint8_t a_state[2]      = { 0, 0};

  for (int i = 0; i < 2; i++) 
  {
    const uint8_t a = digitalRead(a_pins[i]);
    if (a && !a_state[i]) 
    {
      const uint8_t b = digitalRead(b_pins[i]);
      if (b) enc_pulse_counts[pos[i]]--;
      else   enc_pulse_counts[pos[i]]++;
      a_state[i] = 1;
    } 
    else if (!a && a_state[i]) { a_state[i] = 0; }
  }
}
