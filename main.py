import re
import pandas as pd
from PIL import Image
import cv2
import pytesseract
import os  # For file existence check

# Path to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Define entity-unit mapping with constants
entity_unit_map = {
    'width': {'centimetre', 'foot', 'inch', 'metre', 'millimetre', 'yard'},
    'depth': {'centimetre', 'foot', 'inch', 'metre', 'millimetre', 'yard'},
    'height': {'centimetre', 'foot', 'inch', 'metre', 'millimetre', 'yard'},
    'item_weight': {'gram', 'kilogram', 'microgram', 'milligram', 'ounce', 'pound', 'ton'},
    'maximum_weight_recommendation': {'gram', 'kilogram', 'microgram', 'milligram', 'ounce', 'pound', 'ton'},
    'voltage': {'kilovolt', 'millivolt', 'volt'},
    'wattage': {'kilowatt', 'watt'},
    'item_volume': {'centilitre', 'cubic foot', 'cubic inch', 'cup', 'decilitre', 'fluid ounce', 'gallon',
                    'imperial gallon', 'litre', 'microlitre', 'millilitre', 'pint', 'quart'}
}

# Additional common abbreviation fallback mappings
fallback_mapping = {
    'cm': 'centimetre', 'mg': 'milligram', 'kg': 'kilogram', 'g': 'gram',
    'ml': 'millilitre', 'l': 'litre', 'kw': 'kilowatt', 'w': 'watt',
    'mv': 'millivolt', 'kv': 'kilovolt', 'lbs': 'pound'
}

# Load CSV into DataFrame
train_df = pd.read_csv(r"C:\Users\RAHUL\Desktop\test.csv")


# Preprocess image to enhance OCR accuracy
def preprocess_image(image_path):
    # Check if the file exists
    if not os.path.isfile(image_path):
        print(f"Error: File not found at {image_path}")
        return None

    img = cv2.imread(image_path)

    if img is None:
        print(f"Error: Unable to load image at {image_path}")
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    preprocessed_image_path = "preprocessed_image.png"
    cv2.imwrite(preprocessed_image_path, thresh)
    return preprocessed_image_path


# Extract text from image
def extract_text_from_image(image_path):
    return pytesseract.image_to_string(Image.open(image_path))


# Extract numbers and their associated units/words
def extract_numbers_with_units(text):
    pattern = r"(\d+\.?\d*)\s*([a-zA-Z%]+)?"
    matches = re.findall(pattern, text)
    return [f"{num} {unit}" if unit else num for num, unit in matches]


# Get entity name from train.csv based on image filename
def get_entity_name(image_index):
    row_index = image_index - 1
    return train_df.iloc[row_index, 3]  # Assuming entity_name is in the 3rd column


# Map units using abbreviation until unique or fallback if no match
def map_units(extracted_values, entity_name):
    allowed_units = entity_unit_map.get(entity_name, [])

    # Create a mapping of abbreviations from the allowed units (case insensitive)
    unit_map = {}
    for unit in allowed_units:
        abbreviation = unit[:2].lower() if len(unit) > 2 else unit[:1].lower()  # Use 2 letters for longer units, 1 for others
        unit_map[abbreviation] = unit

    # Try to map the extracted values to full units
    best_match = None
    for value in extracted_values:
        number, unit = value.split() if ' ' in value else (value, '')
        unit = unit.lower()  # Convert unit to lowercase for matching

        mapped_unit = None

        # Match only 2 characters for 2-character units (e.g., mg, cm)
        if len(unit) >= 2 and unit[:2] in unit_map:
            mapped_unit = unit_map[unit[:2]]
        elif len(unit) < 2 and unit[:1] in unit_map:
            mapped_unit = unit_map[unit[:1]]

        # If no match is found, use fallback mapping
        if not mapped_unit:
            mapped_unit = fallback_mapping.get(unit)

        if mapped_unit:
            best_match = f"{number} {mapped_unit}"
            break  # Stop once a valid match is found

    return [best_match] if best_match else []


# Convert the mapped value to double format
def convert_to_double(mapped_value):
    if mapped_value:
        number, unit = mapped_value.split()
        try:
            double_value = float(number)
            return f"{double_value:.{len(repr(double_value).split('.')[1])}f} {unit}" if '.' in repr(double_value) else f"{double_value:.1f} {unit}"
        except ValueError:
            return mapped_value
    return ""


# Create an empty list to store the results
results = []

# Loop through all images from index 1 to 100
for image_index in range(1, 101):
    image_filename = f"testimage_{image_index}.jpg"  # Assuming image files are named "testimage_1.jpg", "testimage_2.jpg", etc.
    image_path = rf"C:\Users\RAHUL\Downloads\testimage\{image_filename}"  # Path to the images

    # Step 1: Preprocess the image
    preprocessed_image = preprocess_image(image_path)

    if preprocessed_image:
        # Step 2: Extract text from the preprocessed image
        extracted_text = extract_text_from_image(preprocessed_image)

        # Step 3: Extract numbers and their units from the text
        extracted_values = extract_numbers_with_units(extracted_text)

        # Step 4: Get the entity name from the CSV
        entity_name = get_entity_name(image_index)

        # Step 5: Map the extracted values to the correct units and get the best match
        mapped_values = map_units(extracted_values, entity_name)

        # Step 6: Convert the best match to double format
        double_value = convert_to_double(mapped_values[0] if mapped_values else "")

        # Append the result to the list
        results.append({
            "index": image_index-1,

            "Double Value": double_value
        })

# Convert the results to a DataFrame
results_df = pd.DataFrame(results)

# Save the DataFrame to a CSV file
results_df.to_csv(r"C:\Users\RAHUL\Desktop\extracted_output.csv", index=False)

print("CSV file created successfully!")
