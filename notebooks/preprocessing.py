import pandas as pd
import geopandas as gpd
from fileinput import filename
import pyspark.sql.functions as func


def clean_taxi_data(psdf_taxi, cfg):
    
    
    # we accept > 0 values
    for col in ["passenger_count", "trip_distance", "fare_amount", 
                "total_amount", "duration_in_min"]:
        psdf_taxi = psdf_taxi.withColumn(col, \
                func.when(psdf_taxi[col] > 0, \
                psdf_taxi[col]).otherwise(func.lit(None)))

    # we accept >= 0 values
    for col in ["extra", "tip_amount", "tolls_amount"]:
        psdf_taxi = psdf_taxi.withColumn(col, \
                func.when(psdf_taxi[col] >= 0, \
                psdf_taxi[col]).otherwise(func.lit(None)))

    # values allowed with mod 0.5.    
    psdf_taxi = psdf_taxi.withColumn("extra", \
        func.when(psdf_taxi.extra % 0.5 == 0, \
        psdf_taxi.extra).otherwise(func.lit(None)))

    psdf_taxi = psdf_taxi.withColumn("mta_tax", \
        func.when(psdf_taxi.mta_tax.isin([0, 0.5]), \
        psdf_taxi.mta_tax).otherwise(func.lit(None)))

    psdf_taxi = psdf_taxi.withColumn("improvement_surcharge", \
        func.when(psdf_taxi.improvement_surcharge == 0.3, \
        psdf_taxi.improvement_surcharge).otherwise(func.lit(None)))

    psdf_taxi = psdf_taxi.withColumn("RateCodeID", \
        func.when(psdf_taxi["RateCodeID"].isin([1,2,3,4,5,6]), \
        psdf_taxi["RateCodeID"]).otherwise(func.lit(None)))
    
    for column in ["do_year", "pu_year"]:
        psdf_taxi = psdf_taxi.withColumn(column, \
            func.when(psdf_taxi[column].isin(cfg["extraction"]["yellow_taxi"]["years"]), \
            psdf_taxi[column]).otherwise(func.lit(None)))
    
    for column in ["do_month", "pu_month"]:
        psdf_taxi = psdf_taxi.withColumn(column, \
        func.when(psdf_taxi[column].isin(cfg["extraction"]["yellow_taxi"]["months"]), \
        psdf_taxi[column]).otherwise(func.lit(None)))
        
    columns_to_drop = ['airport_fee', 'congestion_surcharge']
    psdf_taxi = psdf_taxi.drop(*columns_to_drop)

    # We remove those outliers that we detect as probably not correct:
    # - trip_distance values below percentile 1% and above 99.9% are probably not correct
    # - fare_amount values below percentile 0.1% and above 99.9% are probably not correct
    # - tolls_amount values above 99.9% are probably not correct
    # - total_amount values below percentile 0.1% and above 99.9% are probably not correct
    # - duration_in_min values below percentile 0.5% and above 99.5% are probably not correct
    # - tip_amount values above 99.9% are probably not correct

    dict_remove = {
            "trip_distance":[0.01, 0.999],
            "fare_amount":[0.001, 0.999],
            "tolls_amount":[0.0, 0.999],
            "total_amount":[0.001, 0.999],
            "duration_in_min":[0.005, 0.995],
            "tip_amount":[0.0, 0.999],  
            "tip_percentage":[0.0, 0.999],  
        }

    for key_remove in dict_remove:    
        # remove values below percentile        
        psdf_taxi = psdf_taxi.withColumn(key_remove,\
            func.when(func.col(key_remove) < psdf_taxi.agg(\
            func.expr(f'percentile({key_remove}, \
            {dict_remove[key_remove][0]})')).collect()[0][0], func.lit(None)).\
            otherwise(func.col(key_remove)))
        
        # remove values above percentile
        psdf_taxi = psdf_taxi.withColumn(key_remove,\
            func.when(func.col(key_remove) > psdf_taxi.agg(\
            func.expr(f'percentile({key_remove}, \
            {dict_remove[key_remove][1]})')).collect()[0][0], func.lit(None)).\
            otherwise(func.col(key_remove)))
    

    # only keep card payments since other payments does not includ valid tip values
    psdf_taxi = psdf_taxi.filter(psdf_taxi.payment_type == 1)

    return psdf_taxi
    

def add_features_taxi_data(psdf_taxi, psdf_zones, psdf_zones_shp = None):

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
        100 * psdf_taxi.tip_amount / (psdf_taxi.total_amount - psdf_taxi.tip_amount))
    
    # for simplicity just add the temporal feature from dropp off
    psdf_taxi = psdf_taxi.withColumn('day_of_week', 
        func.dayofweek(psdf_taxi.tpep_dropoff_datetime))
    
    psdf_taxi = psdf_taxi.withColumn('day_of_month', 
        func.dayofmonth(psdf_taxi.tpep_dropoff_datetime))
    
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
    
    if psdf_zones_shp != None:
    
        psdf_PU_zones_shp = psdf_zones_shp.select(*(func.col(x).alias('PU_' + x) 
            for x in psdf_zones_shp.columns))
        psdf_DO_zones_shp = psdf_zones_shp.select(*(func.col(x).alias('DO_' + x) 
            for x in psdf_zones_shp.columns))
        
        psdf_taxi = psdf_taxi.join(psdf_PU_zones_shp,
            psdf_taxi["PULocationID"] ==  psdf_PU_zones_shp["PU_OBJECTID"],"left")    

        psdf_taxi = psdf_taxi.join(psdf_DO_zones_shp,
            psdf_taxi["DOLocationID"] ==  psdf_DO_zones_shp["DO_OBJECTID"],"left")    

        psdf_taxi = psdf_taxi.drop("PU_OBJECTID", "DO_OBJECTID")    

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

def add_features_shp_data(gdf_zones_shp):        
    gdf_zones_shp["lat"] = gdf_zones_shp["geometry"].centroid.x
    gdf_zones_shp["lon"] = gdf_zones_shp["geometry"].centroid.y
    return gdf_zones_shp


def show_detailed_percentiles(psdf, column):
    psdf.agg(
        func.expr(f'percentile({column}, array(0.00))')[0].alias('0%'),
        func.expr(f'percentile({column}, array(0.001))')[0].alias('0.1%'),
        func.expr(f'percentile({column}, array(0.005))')[0].alias('0.5%'),
        func.expr(f'percentile({column}, array(0.01))')[0].alias('1%'),
        func.expr(f'percentile({column}, array(0.05))')[0].alias('5%'),
        func.expr(f'percentile({column}, array(0.10))')[0].alias('10%'),
        func.expr(f'percentile({column}, array(0.25))')[0].alias('25%'),
        func.expr(f'percentile({column}, array(0.50))')[0].alias('50%'),
        func.expr(f'percentile({column}, array(0.75))')[0].alias('75%'),
        func.expr(f'percentile({column}, array(0.90))')[0].alias('90%'),
        func.expr(f'percentile({column}, array(0.95))')[0].alias('95%'),
        func.expr(f'percentile({column}, array(0.99))')[0].alias('99%'),
        func.expr(f'percentile({column}, array(0.995))')[0].alias('99.5%'),        
        func.expr(f'percentile({column}, array(0.999))')[0].alias('99.9%'),        
        func.expr(f'percentile({column}, array(1))')[0].alias('100%')
    ).show()    


def show_grouped_percentiles(psdf, group_column, target_column):
    psdf.groupby(group_column).agg(
        func.expr(f'percentile({target_column}, array(0.0))')[0].alias('0%'),                        
        func.expr(f'percentile({target_column}, array(0.05))')[0].alias('5%'),        
        func.expr(f'percentile({target_column}, array(0.25))')[0].alias('25%'),
        func.expr(f'percentile({target_column}, array(0.50))')[0].alias('50%'),
        func.expr(f'percentile({target_column}, array(0.75))')[0].alias('75%'),        
        func.expr(f'percentile({target_column}, array(0.95))')[0].alias('95%'),                
        func.expr(f'percentile({target_column}, array(1))')[0].alias('100%')
    ).orderBy(group_column).show()    