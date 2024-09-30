import sys
import cairosvg
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPM
from PIL import Image
import io

# Check if a file path is provided
if len(sys.argv) < 2:
    print("Please provide the path to the SVG file as an argument.")
    print("Usage: python script_name.py path/to/your/svg_file.svg")
    sys.exit(1)

# Get the file path from command line argument
svg_file_path = sys.argv[1]

# Read the SVG content from the file
try:
    with open(svg_file_path, 'r') as file:
        svg_content = file.read()
except FileNotFoundError:
    print(f"Error: The file {svg_file_path} was not found.")
    sys.exit(1)
except IOError:
    print(f"Error: There was an issue reading the file {svg_file_path}.")
    sys.exit(1)

# Generate output file names based on input file name
base_name = svg_file_path.rsplit('.', 1)[0]
pdf_output = f"{base_name}.pdf"
png_output = f"{base_name}.png"

# Convert SVG to PDF
try:
    cairosvg.svg2pdf(bytestring=svg_content, write_to=pdf_output)
    print(f"PDF saved as '{pdf_output}'")
except Exception as e:
    print(f"Error converting to PDF: {e}")

# Convert SVG to PNG
try:
    drawing = svg2rlg(io.StringIO(svg_content))
    png_data = renderPM.drawToString(drawing, fmt="PNG", dpi=72) # 144 for high resolution

    # Open the PNG data with Pillow and save it
    img = Image.open(io.BytesIO(png_data))
    img.save(png_output)
    print(f"PNG saved as '{png_output}'")
except Exception as e:
    print(f"Error converting to PNG: {e}")

print("Conversion complete.")