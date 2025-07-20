import os
import zipfile
import shutil


dirs = [
    'configs',
    'results',
    'models',
    'data'
]

for d in dirs:
    os.makedirs(d, exist_ok=True)


if os.path.exists('step_dpo_model.zip'):
    with zipfile.ZipFile('step_dpo_model.zip', 'r') as zip_ref:
        zip_ref.extractall('models/')
    print("Unzipped models.")


if os.path.exists('step_dpo_data.zip'):
    with zipfile.ZipFile('step_dpo_data.zip', 'r') as zip_ref:
        zip_ref.extractall('data/')
    print("Unzipped data.")


if os.path.exists('results.txt'):
    shutil.move('results.txt', 'results/results.txt')
    print("Moved results.txt.")

print("Setup complete. Delete zips manually if desired.")