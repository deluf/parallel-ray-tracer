import os
import shutil
from datetime import datetime
import time

# Define the destination path
destination_base = "simulations/"

# Create a folder with the current timestamp
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
destination_folder = os.path.join(destination_base, timestamp)

os.makedirs(destination_folder, exist_ok=True)
print(f"Created folder: {destination_folder}")

print(f"Running make...")
os.system("make clean")
os.system("make")
print("")

for i in range(1, 13):

    # Run the raytracer.exe with the current iteration and redirect output
    output_file = destination_folder + f"/stdout-{i}.txt"
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} > Running ./raytracer {i} ...", end=" ")

    start_time = time.time()
    os.system(f"./raytracer.exe {i} > {output_file}")
    elapsed_time = (time.time() - start_time) * 1000.0

    print(f"{elapsed_time:.3f} ms")

# Copy the render.bmp file to the destination folder
shutil.copy("render.bmp", destination_folder)

print(f"Done!")
