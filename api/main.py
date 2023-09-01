import asyncio
import itertools
import os
import time
from functools import lru_cache
from typing import Tuple

import aiohttp
import mlflow.pyfunc
import pandas as pd
from fastapi import FastAPI, File, UploadFile
# from pyspark.sql import SparkSession
from geopy import distance
from models import HouseDetails

app = FastAPI(swagger_ui_parameters={"tryItOutEnabled": True})

MODEL_NAME, STAGE = "CatBoostModel", "Production"


primary_schools_df = pd.read_csv(
    os.path.join(os.path.dirname(__file__), "data/primary_schools.csv")
)
mrt_stations_df = pd.read_csv(
    os.path.join(os.path.dirname(__file__), "data/mrt_stations.csv")
)
RAFFLES_PLACE_LAT, RAFFLES_PLACE_LONG = 1.284184, 103.85151

# spark = SparkSession.builder.getOrCreate()

@lru_cache(maxsize=None)
def calculate_geodesic_dist(
    start_lat: float, start_long: float, dest_lat: float, dest_long: float
):
    return distance.distance((start_lat, start_long), (dest_lat, dest_long)).km


@lru_cache(maxsize=None)
def calculate_dist_to_nearest_primary_school(start_lat: float, start_long: float):
    distances = primary_schools_df.apply(
        lambda row: calculate_geodesic_dist(
            start_lat, start_long, row["lat"], row["long"]
        ),
        axis=1,
    )
    return min(distances)


@lru_cache(maxsize=None)
def calculate_dist_to_nearest_mrt_station(start_lat: float, start_long: float):
    distances = mrt_stations_df.apply(
        lambda row: calculate_geodesic_dist(
            start_lat, start_long, row["lat"], row["long"]
        ),
        axis=1,
    )
    return min(distances)


async def async_get_coordinates(address: str) -> Tuple[float, float]:
    headers = {"Authorization": os.environ["ACCESS_TOKEN"]}
    url = f"https://www.onemap.gov.sg/api/common/elastic/search?searchVal={address}&returnGeom=Y&getAddrDetails=Y"
    async with aiohttp.ClientSession(headers=headers) as session:
        response = await session.get(url)
        while response.status == 429:
            time.sleep("10")
            response = await session.get(url)

        try:
            response_json = await response.json()
            result = response_json["results"][0]
            return (result["LATITUDE"], result["LONGITUDE"])
        except Exception as e:
            return (None, None)

async def limit_openmapapi_calls(df: pd.DataFrame, func):
    df["house_lat"] = None
    df["house_long"] = None
    processes = [func(index, row) for index, row in df.iterrows()]
    it = iter(processes)
    while True:
        batch = list(itertools.islice(it, 500))
        if not batch:
            break
        for result in await asyncio.gather(*batch):
            row_id, lat, long = result
            df.loc[row_id, "house_lat"] = lat
            df.loc[row_id, "house_long"] = long
        time.sleep(5)


@app.post("/predict/")
async def predict(house_details: HouseDetails):
    house_lat, house_long = await async_get_coordinates(house_details.address)
    distance_cbd = calculate_geodesic_dist(
        house_lat, house_long, RAFFLES_PLACE_LAT, RAFFLES_PLACE_LONG
    )
    distance_top_primary = calculate_dist_to_nearest_primary_school(
        house_lat, house_long
    )
    distance_mrt = calculate_dist_to_nearest_mrt_station(house_lat, house_long)
    model = mlflow.pyfunc.load_model(model_uri=f"models:/{MODEL_NAME}/{STAGE}")
    input = pd.DataFrame(
        {
            "town": house_details.town,
            "flat_type": house_details.flat_type,
            "storey_range": house_details.storey_range,
            "floor_area_sqm": house_details.floor_area_sqm,
            "flat_model": house_details.flat_model,
            "remaining_lease": house_details.remaining_lease,
            "distance_cbd": distance_cbd,
            "distance_top_primary": distance_top_primary,
            "distance_mrt": distance_mrt,
        },
        index=[0],
    )
    print(input)
    output = model.predict(input)[0]
    print(output)
    return {"resale per sqm": output, "resale": output * house_details.floor_area_sqm}


@app.post("/batch_predict/")
async def batch_predict(csv_file: UploadFile = File(...)):
    df = pd.read_csv(csv_file.file)
    
    # Convert resale price to resale price per sqm
    df["resale_price_per_sqm"] = df.resale_price / df.floor_area_sqm
    df = df.drop(columns="resale_price")

    # Convert month, flat_type, storey_range, flat_model to categorical values
    df.month = pd.Categorical(df.month)
    df.town = pd.Categorical(df.town)
    df.flat_type = pd.Categorical(df.flat_type)
    df.storey_range = pd.Categorical(df.storey_range)
    df.flat_model = pd.Categorical(df.flat_model)

    # Set types
    df["block"] = df["block"].astype("string")
    df["street_name"] = df["street_name"].astype("string")
    print(df.head())
    # await limit_openmapapi_calls(df, async_get_coordinates)
    # df = df.dropna(subset=["house_lat", "house_long"])

    # # Get CBD travel distance
    # df["distance_cbd"] = df.apply(lambda row: calculate_geodesic_dist(row["house_lat"], row["house_long"], RAFFLES_PLACE_LAT, RAFFLES_PLACE_LONG), axis=1)

    # # Get distance to nearest popular primary school
    # df["distance_top_primary"] = df.apply(lambda row: calculate_dist_to_nearest_primary_school(row["house_lat"], row["house_long"]), axis=1)

    # # Get distance to nearest MRT
    # df["distance_mrt"] = df.apply(lambda row: calculate_dist_to_nearest_mrt_station(row["house_lat"], row["house_long"]), axis=1)

    # columns_to_drop =[
    #         "block",
    #         "street_name",
    #         "lease_commence_date",
    #         "resale_price_per_sqm",
    #         "house_lat",
    #         "house_long",
    #         "month",
    #     ] 

    # input_df = df.drop(columns=columns_to_drop)
    # spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
    # spark_df = spark.createDataFrame(input_df) 
    # result_pdf = spark_df.select("*").toPandas()
    # print(result_pdf.head())
    # pyfunc_udf = mlflow.pyfunc.spark_udf(spark, f"models:/{MODEL_NAME}/{STAGE}")
    # output_df = spark_df.withColumn("prediction", pyfunc_udf())
    # return output_df
