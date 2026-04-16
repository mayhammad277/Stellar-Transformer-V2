import numpy as np
from sklearn.cluster import KMeans
import json
import csv

# === Step 1: Load RA/Dec from file ===
ra_list = []
dec_list = []


ra_list2 = []
dec_list2 = []
filenames=[]
with open('/home/student/star_tracker/ra_dec_from_images_new.csv', 'r') as csvfile:  # Replace with your actual path
    reader = csv.DictReader(csvfile)
    num_v=5001
    for row in reader:
        ra_list2.append(float(row['RA']))
        dec_list2.append(float(row['Dec']))
        u=row['Filename'].split("_")
        v=row['Filename'].split(".")
        f_new=u[0]+"-"+str(num_v)+".png"

        filenames.append(f_new)
        num_v+=1
with open('/media/student/B076126976123098/my_data/SiT/dataset_sky/ra-dec.txt', 'r') as f:
    for line in f:
        ra_str, dec_str = line.strip().split('_')
        ra_list.append(float(ra_str))
        dec_list.append(float(dec_str))
ra_list_t=ra_list+ra_list2
dec_list_t=dec_list+dec_list2
ra_list_t = np.array(ra_list_t)
dec_list_t = np.array(dec_list_t)

# === Step 2: Convert to radians ===
ra_rad = np.deg2rad(ra_list_t)
dec_rad = np.deg2rad(dec_list_t)

# === Step 3: Convert RA/Dec to 3D unit vectors ===
x = np.cos(dec_rad) * np.cos(ra_rad)
y = np.cos(dec_rad) * np.sin(ra_rad)
z = np.sin(dec_rad)
coords_3d = np.stack((x, y, z), axis=1)

# === Step 4: KMeans clustering ===
k = 12  # Number of orientation classes
kmeans = KMeans(n_clusters=k, random_state=42)
labels = kmeans.fit_predict(coords_3d)

# === Step 5: Save cluster centers as RA/Dec ===
def cartesian_to_radec(x, y, z):
    dec = np.arcsin(z)
    ra = np.arctan2(y, x)
    ra = np.rad2deg(ra) % 360
    dec = np.rad2deg(dec)
    return ra, dec

cluster_centers = kmeans.cluster_centers_
ra_centers = []
dec_centers = []

for center in cluster_centers:
    ra_c, dec_c = cartesian_to_radec(*center)
    ra_centers.append(ra_c)
    dec_centers.append(dec_c)

# Save cluster centers to JSON
centers_data = [
    {"class_id": i, "ra_deg": ra_centers[i], "dec_deg": dec_centers[i]}
    for i in range(k)
]

with open("orientation_class_centers_new.json", "w") as f:
    json.dump(centers_data, f, indent=2)

print("Cluster centers saved to orientation_class_centers.json")

# === Step 6: Save pseudo-labels for each image ===
with open('image_labels_new_2.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['image_name', 'ra_deg', 'dec_deg', 'orientation_class'])

    for i, (ra, dec, label) in enumerate(zip(ra_list, dec_list, labels[:len(dec_list)])):
        image_name = f"stars-{i:03d}.png"  # Adjust as needed
        writer.writerow([image_name, ra, dec, label])

    for i, (ra, dec, label,image_name) in enumerate(zip(ra_list2, dec_list2, labels[len(dec_list):],filenames)):
             

        writer.writerow([image_name, ra, dec, label])

print("Image labels saved to image_labels.csv")

