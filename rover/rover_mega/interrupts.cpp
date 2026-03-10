/**
 * interrupts.cpp
 * Created 9/6/2025
 * 
 * Masks interrupt registers on ATmega2560 MPU and routes ISRs to 
 * relevant functions.
 */

#include <Arduino.h>
#include "config.h"
#include "interrupts.h"

// Handlers are implemented in their respective modules
void motors_handle_pcint0_encoders(); // PB4-PB7
void motors_handle_pcint1_encoders(); // PJ0-PJ1
void sensors_handle_pcint0_echoes();  // PB0-PB2

/*
Configures Arduino interrupt registers with masks to enable both
overall port interrupts (PCICR) and individual pin interrupts
(PCMSKn) in accordance with connected subsystems. 
*/
void interrupts_setup() 
{
  uint8_t pcicr_mask = 0;
  uint8_t pcmsk0_mask = 0;
  uint8_t pcmsk1_mask = 0;

  if (motors_attached)
  {
    pcicr_mask  |= 0x03;     // PCIE0 (PB0-7) + PCIE1 (PJ0-6)
    pcmsk0_mask |= 0xF0;     // PB4-PB7 encoders
    pcmsk1_mask |= 0x06;     // PJ0-PJ1 encoders
  }
  if (ultrasonics_attached)
  {
    pcicr_mask |= 0x01;      // PCIE0 (PB0-7)
    pcmsk0_mask |= 0x07;     // PB0-PB2 echoes
  }

  // Apply masks to ATmega2560 registers, see datasheet
  PCICR  |= pcicr_mask;
  PCMSK0 |= pcmsk0_mask;
  PCMSK1 |= pcmsk1_mask;
}

/*
Interrupt handler for PCINT0 register (PB0-7).
*/
ISR(PCINT0_vect)
{
  if (motors_attached)      motors_handle_pcint0_encoders();
  if (ultrasonics_attached) sensors_handle_pcint0_echoes();
}

/*
Interrupt handler for PCINT1 register (PJ0-6).
*/
ISR(PCINT1_vect)
{
  if (motors_attached)      motors_handle_pcint1_encoders();
}
