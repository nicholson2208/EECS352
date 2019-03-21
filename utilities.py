def create_wav_file_mapping(file_name):
    """
    
    returns a dictionary with numbers (as strings) as the key and a dictionary with {"name" : "a", "instrument" : "piano"} 
    """
    
    # Assuming the format "number A_instrument"
    mapping = dict()
    
    with open(file_name, 'r') as f:
        lines = f.readlines()
        
        for line in lines:            
            broken_line = line.split()
            
            chord_num = broken_line[0]
            
            chord_name, instrument = broken_line[1].split("_")
            
            chord_dict = {"name" : chord_name, "instrument" : instrument}
            
            mapping[str(chord_num)] = chord_dict
        
    
    return mapping
    

def get_chord_name(chord_list_numbers, mapping):
    """
    
    returns : (chord_name_list, instrument_name_list)
    """
    chord_name_list = []
    instrument_name_list = []
       
    
    for chord_nums in chord_list_numbers:
        chord_name_list.append(mapping[str(chord_nums)]["name"])
        instrument_name_list.append(mapping[str(chord_nums)]["instrument"])
    
    
    return chord_name_list, instrument_name_list