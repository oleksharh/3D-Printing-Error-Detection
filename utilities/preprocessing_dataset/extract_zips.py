import zipfile

for i in range(0, 192):
    path_to_zip_file = f'S:/raw_data/print{i}.zip'
    directory_to_extract_to = f'C:/FYP/full_dataset/'

    print(f'Extracting {path_to_zip_file} to {directory_to_extract_to}...', end='\r')
    with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
        zip_ref.extractall(directory_to_extract_to)
    print(f'Finished extracting {path_to_zip_file}.', end='\r')