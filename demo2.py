import os


# instead of a single folder, make different folder for each person
# align the faces before Saving
os.system("python dataset_creator2.py")

# after different folders are created for each student then code needs to be changed
os.system("python training_set.py")

# make an array of all the students in the database initialied as zero
# Add alignment of faces before recognitioin
# keep a single picture with squares instead of multiple faces in different folder
os.system("python recognize2.py")
