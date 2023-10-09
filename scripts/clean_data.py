import pandas as pd
import glob 
import os 
import shutil

path_list = ['dataset/argoverse/train/**/*.parquet','dataset/argoverse/val/**/*.parquet']

# path_list = ['dataset/argoverse/test/**/*.parquet']
for plist in path_list:

    paths = glob.glob(plist)
    index = 0 
    for path in paths:
        df = pd.read_parquet(path)
        agent = df[df['track_id']==df['focal_track_id'][0]]
        len_agent = len (agent)
        if len_agent < 110:
            dir_name = os.path.dirname(path)
            print(dir_name)
            shutil.rmtree(dir_name)
            # os.rmdir(dir_name)   # only for empty folder.
            # os.remove(path)   #only for single file.
            index += 1 
            print(index , " ", len_agent)
    print("The number of invalid agents is: ", index)