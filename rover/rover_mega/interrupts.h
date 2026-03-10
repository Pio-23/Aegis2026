/**
 * interrupts.h
 * Created 9/6/2025
 * 
 * Masks interrupt registers on ATmega2560 MPU and routes ISRs to relevant 
 * functions.
 */

#ifndef AEGIS_INTERRUPTS_H
#define AEGIS_INTERRUPTS_H
#include <Arduino.h>

void interrupts_setup();

#endif
