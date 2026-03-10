/**
 * uart.h
 * Created 9/6/2025
 * 
 * Handles both the reception and execution of Raspberry Pi commands and the
 * composition and transmission of telemetry strings over Serial1.
 */

#ifndef AEGIS_UART_H
#define AEGIS_UART_H
#include <Arduino.h>

void uart_do_command();
void uart_send_telemetry();

#endif
