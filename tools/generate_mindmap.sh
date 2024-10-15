#!/bin/bash

# npm install markmap-lib markmap-render node-html-to-image

# Check if an input file is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <input_markdown_file>"
    exit 1
fi

# Get the input file name
input_file="$1"

# Check if the input file exists
if [ ! -f "$input_file" ]; then
    echo "Error: Input file '$input_file' does not exist."
    exit 1
fi

# Generate the output file name
output_file="${input_file}.png"

# Run the Node.js script
node markmap-to-png.js "$input_file" "$output_file"

# Check if the Node.js script was successful
if [ $? -eq 0 ]; then
    echo "Mindmap generated successfully: $output_file"
else
    echo "Error: Failed to generate mindmap."
    exit 1
fi
