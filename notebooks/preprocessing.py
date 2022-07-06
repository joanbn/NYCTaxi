import pandas as pd
import geopandas as gpd
from fileinput import filename
import pyspark.sql.functions as func

def clean_taxi_data(psdf_taxi, cfg):

    # count = psdf_taxi.count()

    psdf_taxi = psdf_taxi.filter(psdf_taxi.passenger_count > 0)
    # passenger_count is a manual annotation by the rider, so it can be error prone
    psdf_taxi = psdf_taxi.filter(psdf_taxi.passenger_count <= 8)
    psdf_taxi = psdf_taxi.filter(psdf_taxi.trip_distance > 0)
    psdf_taxi = psdf_taxi.filter(psdf_taxi.fare_amount > 0)        
    psdf_taxi = psdf_taxi.filter(psdf_taxi.extra >= 0) # we accept extra = 0
        
    # Be aware that a change of allowed values filtered below
    # might cause to discard valid option in the future    
    psdf_taxi = psdf_taxi.filter(psdf_taxi.extra % 0.5 == 0) # values allowed with mod 0.5.    
    psdf_taxi = psdf_taxi.filter(psdf_taxi.mta_tax.isin([0, 0.5]))    
    psdf_taxi = psdf_taxi.filter(psdf_taxi.improvement_surcharge == 0.3) 
    psdf_taxi = psdf_taxi.filter(psdf_taxi.total_amount > 0)     
    psdf_taxi = psdf_taxi.filter(psdf_taxi.tip_amount >= 0) # we accept top = 0
    psdf_taxi = psdf_taxi.filter((psdf_taxi["RateCodeID"]>=1) & (psdf_taxi["RateCodeID"]<=6))         

    psdf_taxi = psdf_taxi.filter(psdf_taxi["sum_amount"] == psdf_taxi["total_amount"])

    psdf_taxi = psdf_taxi.filter(psdf_taxi["tpep_pickup_datetime"] < psdf_taxi["tpep_dropoff_datetime"])    
    
    psdf_taxi = psdf_taxi.filter(
        psdf_taxi["do_year"].isin((cfg["extraction"]["yellow_taxi"]["years"])) 
        | (psdf_taxi["pu_year"].isin(cfg["extraction"]["yellow_taxi"]["years"])))
        
    psdf_taxi = psdf_taxi.filter(
        psdf_taxi["do_month"].isin((cfg["extraction"]["yellow_taxi"]["months"])) 
        & (psdf_taxi["pu_month"].isin(cfg["extraction"]["yellow_taxi"]["months"])))

    psdf_taxi = psdf_taxi.filter(psdf_taxi.duration_in_min>=0)

    vars_to_remove_outliers = ['duration_in_min']

    for var_remove in vars_to_remove_outliers:
        df_stats = psdf_taxi.select(
            func.mean(func.col(var_remove)).alias('mean'),
            func.stddev(func.col(var_remove)).alias('std'),
            func.max(func.col(var_remove)).alias('max')
        ).collect()

        mean = df_stats[0]['mean']
        std = df_stats[0]['std']
        max = df_stats[0]['max']    
        max_outlier = mean + 3 * std        

        psdf_taxi = psdf_taxi.filter(psdf_taxi[var_remove] <= max_outlier)        

    # drop cash payments since we don't have tip information
    psdf_taxi = psdf_taxi.filter(psdf_taxi.payment_type != 2)

    # dropped_cash_rows = filtered_count - psdf_taxi.count()
    # print(f"Dropped cash type rows: {dropped_cash_rows} ({round(100*dropped_cash_rows/count,2)}%)")

    return psdf_taxi
    

def add_features_taxi_data(psdf_taxi, psdf_zones):

    psdf_taxi = psdf_taxi.withColumn('sum_amount', 
            func.round(psdf_taxi.tip_amount + psdf_taxi.fare_amount + 
            psdf_taxi.extra + psdf_taxi.mta_tax + 
            psdf_taxi.improvement_surcharge + psdf_taxi.tolls_amount, 2))

    psdf_taxi = psdf_taxi.withColumn('pu_month', func.month(psdf_taxi.tpep_pickup_datetime))
    psdf_taxi = psdf_taxi.withColumn('pu_year', func.year(psdf_taxi.tpep_pickup_datetime))
    psdf_taxi = psdf_taxi.withColumn('do_month', func.month(psdf_taxi.tpep_dropoff_datetime))
    psdf_taxi = psdf_taxi.withColumn('do_year', func.year(psdf_taxi.tpep_dropoff_datetime))

    psdf_taxi = psdf_taxi.withColumn('duration_in_min', 
        func.round((psdf_taxi["tpep_dropoff_datetime"].cast("long") 
        - psdf_taxi["tpep_pickup_datetime"].cast("long"))/60,1))        

    psdf_taxi = psdf_taxi.withColumn('tip_percentage', \
        100 * psdf_taxi.tip_amount / psdf_taxi.total_amount)
    
    # for simplicity just add the temporal feature from dropp off
    psdf_taxi = psdf_taxi.withColumn('day_of_week', 
        func.dayofweek(psdf_taxi.tpep_dropoff_datetime))
    psdf_taxi = psdf_taxi.withColumn('hour', 
        func.hour(psdf_taxi.tpep_dropoff_datetime))
    
    psdf_PU_zones = psdf_zones.select(*(func.col(x).alias('PU_' + x) 
        for x in psdf_zones.columns))
    psdf_DO_zones = psdf_zones.select(*(func.col(x).alias('DO_' + x) 
        for x in psdf_zones.columns))

    psdf_taxi = psdf_taxi.join(psdf_PU_zones,
        psdf_taxi["PULocationID"] ==  psdf_PU_zones["PU_LocationID"],"left")    

    psdf_taxi = psdf_taxi.join(psdf_DO_zones,
        psdf_taxi["DOLocationID"] ==  psdf_DO_zones["DO_LocationID"],"left")    

    psdf_taxi = psdf_taxi.drop("PU_LocationID", "DO_LocationID")

    # TODO
    #   - add holidays
    #   - 

    return psdf_taxi

def clean_zone_data(df_zones):
    df_zones = df_zones.fillna('Unknown')
    return df_zones


def clean_zone_shp_data(gdf_zones_shp):    
    # we saw that LocationID has duplicates so we will use OBJECTID instead
    gdf_zones_shp = gdf_zones_shp.drop(columns=['LocationID'])
    return gdf_zones_shp
