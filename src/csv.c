#include "csv.h"

float32_t power(float32_t i, int8_t pow) {
	float32_t ret = i;
	if(pow > 0) {
		for(int8_t n = 1; n < pow; n++) { ret *= i; }
	}
	else if(pow < 0) {
		for(int8_t n = -1; n >= pow; n--) { ret /= i; }
	}
	else { ret = 1.0; }
	//printf("\t\t\t\t\tpow(%f, %d) = %f\r", i, pow, ret);

	return ret;
}

enum parse_state_enum { start, integer, fractional, exponent_state, next };
typedef enum parse_state_enum parse_state_t;

int matrixFromCSV(const char *path, size_t height, size_t width, matrix32f_t *mat) {
    // Load file
    FILE *csvFile = fopen(path, "r");
    if(csvFile == NULL) { return 30; }
    char *buffer = malloc(sizeof(char) * BUFFERSIZE);

	// Set to 1 if the end of the file was reached. EOF is set during parsing, NOT READING
	uint8_t eof = 0;
	// Set if an error occurs
	int8_t err = 0;
	// Buffer to store floats
	float32_t *tempf;

    // Create matrix
    if(height==0 || width== 0) { err = 1; goto exception; }

    mat->h = height;
    mat->w = width;
	size_t alloc_floats = mat->h * mat->w;
	tempf = (float32_t*)malloc(alloc_floats * sizeof(float32_t));

	// Create variables for handling the read buffer
	// Increased while parsing; shows when a new read should occur
	size_t buffer_idx = 0;
	// The number of valid bytes in buffer; When buffer_idx == buffer_len, a read occurs
	size_t buffer_len = 0; // set to 0 to force a read

    float32_t f;
	char c = 1;
	int8_t exponent, negative_exponent;
	uint8_t is_negative, pos_after_point;
	size_t floats_read = 0;
	parse_state_t parse_state = start;
	while(!eof) {
		//printf("State: %d, f=%2.6f, buffer[%d]=%c%s\n", parse_state, f, buffer_idx, buffer[buffer_idx], (buffer[buffer_idx]?"":" (EOF)"));
		// Fill buffer if required
		if(buffer_idx == buffer_len && c != 0x00) {
			buffer_idx = 0;
			buffer_len = fread(buffer, 1, BUFFERSIZE, csvFile);
			if(buffer_len < BUFFERSIZE ) { buffer[buffer_len] = 0x00; }
			//printf("Read %d bytes\n", buffer_len);
		}

		// process one character
		switch(parse_state){
			// State for starting parsing
			case start:
				if(buffer[buffer_idx] == 0x00) { eof = 1; break; }
				f = 0.00;
				is_negative = 0;
				exponent = 0;
				negative_exponent = 0;
				pos_after_point = 0;
				parse_state = integer;
				break;

			// State entered before any special character
			case integer:
				c = buffer[buffer_idx];
				if(c >= '0' && c <= '9'){ f = f*10 + (int8_t)(c-'0'); } // got new digit
				else if(c == '-') { is_negative = 1; }	// got a sign
				else if(c == '+') { is_negative = 0; }
				else if(c == '.') { parse_state = fractional; } // time to go to fractional part
				else if(c == 'e' || c == 'E') { parse_state = exponent_state; } // time to read the exponent
				else if(c == ',' || c=='\n' || c == 0x00) { parse_state = next; if(c==0x00) {break;} } // time to wrap-up this float
				else { err = 1; goto exception; } // everything else is an error
				buffer_idx++;
			break;

			// State entered after after reading '.'
			// This character may not be present on all floats
			case fractional:
				c = buffer[buffer_idx];
				if(c >= '0' && c <= '9'){ // got new digit
					pos_after_point++;
					f += ((float)(c-'0')) / power(10.0, pos_after_point);
				}
				else if(c == 'e' || c == 'E')  { parse_state = exponent_state;} // time to get exponent
				else if(c == ',' || c=='\n' || c == 0x00) { parse_state = next; if(c==0x00) {break;} } // time to wrap-up this float
				else { err = 2; goto exception; } // everything else is an error
				buffer_idx++;
			break;

			// State entered when ready to read the exponent
			case exponent_state:
				c = buffer[buffer_idx];
				if(c >= '0' && c <= '9'){ // got new digit
					exponent = exponent*10 + ((int8_t)(c-'0'));
				}
				else if(c == '-') { negative_exponent = 1; }
				else if(c == '+') { negative_exponent = 0; }
				else if(c == ',' || c=='\n' || c == 0x00) { // time to wrap up
					if(negative_exponent) { exponent = -1*(exponent+1); }
					f *= power(10.0, exponent);
					parse_state = next;
				}
				else { err = 2; goto exception; }
				buffer_idx++;
			break;

			// State entered when a float has been parsed
			case next:
				f *= (is_negative) ? -1.0 : 1.0;
				// Check we haven't read too many floats before writing a new one
				if(floats_read < alloc_floats){
					// Store float
					tempf[floats_read] = f;
					floats_read++;
				}
				else { err = 80; goto exception; }

				// Ready to get new float
				parse_state = start;
				// Set eof if needed
				eof = (c == 0x00);
				//if(eof) { printf("EOF reached!\n"); }

			break;

			default:
				err = 100;
				goto exception;
			break;
		}
	}

	// Check if we read as many floats as we anticipated
	if(floats_read != mat->h * mat->w) {
		err = 127;
		printf("Error: Read %d floats, not %d!\n", floats_read,  mat->h * mat->w);
		goto exception;
	}

	// Wrap-up matrix
	mat->d = tempf;
	fclose(csvFile);
	free(buffer);
	return 0;

exception:
	//mat->d = NULL;
	if(err != 1) { free(tempf); }
	free(buffer);
    fclose(csvFile);
	return err;
}
