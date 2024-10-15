

def initialise_variables():

    with open('variable_config.txt', 'r') as file:
        file_content = file.read()

    def extract_section(content, header):  # Function to extract values under each section header
        start = content.find(header)  # Locate the start of the section by finding the header
        if start == -1:
            return []

        section_content = content[start + len(header):].strip()  # Extract the part after the header
        lines = section_content.split('\n')

        values = []
        for line in lines:
            stripped_line = line.strip()
            if stripped_line == '':
                continue
            if '=' in stripped_line:
                break
            values.append(stripped_line)

        return values

    extract_properties = extract_section(file_content, 'extract_properties =')
    headers = extract_section(file_content, 'headers =')

    def extract_id(content):

        combined_ids = []

        for line in content:
            if '=' in line:
                var_list = eval(line.split('=')[1].strip())
                combined_ids.extend(var_list)

        return combined_ids

    with open('elements.txt', 'r') as element_file:
        element_content = element_file.readlines()

    element_ids = extract_id(element_content)

    with open('compounds.txt', 'r') as compound_file:
        compound_content = compound_file.readlines()

    compound_ids = extract_id(compound_content)

    return element_ids, compound_ids, extract_properties, headers
