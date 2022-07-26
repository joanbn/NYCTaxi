import pyarrow.parquet as pq
import zipfile
import pandas as pd
import geopandas as gpd
from fileinput import filename
from urllib import request
import os.path
import gc

from pyspark.sql.types import StructType

RELATIVE_ROOT_PATH = ".."

def get_taxi_data(spark, cfg):

    # df_response = spark.createDataFrame([], StructType([]))
    df_response = None
    
    for year in cfg["extraction"]["yellow_taxi"]["years"]:
        for month in cfg["extraction"]["yellow_taxi"]["months"]:
            
            file_name = cfg["extraction"]["yellow_taxi"]["file_name"]\
                .replace("$YEAR", str(year)) \
                .replace("$MONTH", str(month).zfill(2))

            local_file = RELATIVE_ROOT_PATH + cfg["extraction"]["yellow_taxi"]["local_folder"] + file_name
            
            if os.path.exists(local_file):
                print(f"Already downloaded file: {file_name}")
            else:        
                url = cfg["extraction"]["yellow_taxi"]["url"] + file_name
                print(f"Downloading file: {file_name}")        
                request.urlretrieve(url, local_file)
                print("Done")

            df = spark.read.parquet(local_file)            
            df_sample = df.sample(cfg["extraction"]["yellow_taxi"]["sample_fraction"])
            del df
            gc.collect()

            # create an empty dataframe if it's the first iteration
            if df_response is None:
                emp_RDD = spark.sparkContext.emptyRDD()
                df_response = spark.createDataFrame(data=emp_RDD,
                                    schema=df_sample.schema)               

            df_response = df_response.union(df_sample)        

            del df_sample
            gc.collect()

            

    return df_response

def get_zones(cfg):
    file_name = cfg["extraction"]["zones_table"]["file_name"]        
    local_file = RELATIVE_ROOT_PATH + cfg["extraction"]["zones_table"]["local_folder"] + file_name

    if os.path.exists(local_file):
        print(f"Already downloaded file: {file_name}")
    else:        
        url = cfg["extraction"]["zones_table"]["url"] + file_name
        print(f"Downloading file: {file_name}")        
        request.urlretrieve(url, local_file)
        print("Done")         

    return pd.read_csv(local_file)

def get_zones_shapefile(cfg):    
    
    file_name_wo_extension = cfg["extraction"]["zones_shapefile"]["file_name_wo_extension"]    
    local_folder_shp = RELATIVE_ROOT_PATH \
        + cfg["extraction"]["zones_shapefile"]["local_folder"] \
        + file_name_wo_extension
    
    local_file_shp = local_folder_shp + "/" + file_name_wo_extension + ".shp"

    if os.path.exists(local_file_shp):        
        print(f"Already downloaded file: {file_name_wo_extension}.shp")
    else:        
        file_name = cfg["extraction"]["zones_shapefile"]["file_name"]
        url = cfg["extraction"]["zones_shapefile"]["url"] + file_name            
        local_file = RELATIVE_ROOT_PATH + cfg["extraction"]["zones_shapefile"]["local_folder"] + file_name
        print(f"Downloading file: {file_name}")        
        request.urlretrieve(url, local_file)
        print("Done") 
        
        if not os.path.exists(local_folder_shp):            
            os.makedirs(local_folder_shp)                

        print(f"Unzipping file: {file_name}")        
        with zipfile.ZipFile(local_file,"r") as zip_ref:
            zip_ref.extractall(local_folder_shp)
               
    return gpd.read_file(local_file_shp)
    