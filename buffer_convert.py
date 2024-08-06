import csv
import struct

def hex_to_float(hex_str):
    # Convert the hex string to a float
    return struct.unpack('!f', bytes.fromhex(hex_str))[0]

def process_csv(input_file, output_file, delimiter=", "):
    with open(input_file, mode='r') as infile, open(output_file, mode='w', newline='') as outfile:
        # Write header for the output CSV
        writer = csv.writer(outfile)
        writer.writerow(['data'])
        
        for line in infile:
            # Split the line using the multi-character delimiter
            row = line.strip().split(delimiter)
            
            # Skip the header row
            if row[0] == 'Element':
                continue

            # Convert data.x and data.z from hex to float
            data_x = hex_to_float(row[1])
            
            data_z = hex_to_float(row[3])
            
            
            # data_x = int(row[1],16)
            # data_y = int(row[2],16)
            # data_z = int(row[3],16)
            # data_w = int(row[4],16)

            # Write the values in separate rows in the output file
            writer.writerow([data_x])
            # writer.writerow([data_y])
            writer.writerow([data_z])
            # writer.writerow([data_w])

input_file = 'hadamard_shader.csv'  # Replace with your input file path
output_file = 'hadamard_outptu_converted.csv'  # Replace with your desired output file path

process_csv(input_file, output_file)