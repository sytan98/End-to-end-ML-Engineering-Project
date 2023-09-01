import asyncio
import io
import itertools
import logging
import time
from datetime import datetime
from dateutil.relativedelta import relativedelta
from functools import lru_cache
from typing import List, Tuple
import os

import aiohttp
import pandas as pd
import requests
from geopy import distance
from requests import request
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

primary_schools_df = pd.read_csv(os.path.join(os.path.dirname(__file__), "data/primary_schools.csv"))
mrt_stations_df =  pd.read_csv(os.path.join(os.path.dirname(__file__), "data/mrt_stations.csv"))
RAFFLES_PLACE_LAT, RAFFLES_PLACE_LONG = 1.284184, 103.85151


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

def replace_address_abbreviation(address: str) -> str:
    malay_mapping = {"JLN": "JALAN", "BT": "BUKIT", 
                     "KG": "KAMPONG", "Lor": "LORONG",
                     "TG": "TANJONG"}
    for key, value in malay_mapping.items():
        if key in address:
            address = address.replace(key, value)
    return address
    
async def async_get_coordinates(address: str) -> Tuple[float, float]:
    headers = {"Authorization": os.environ["ACCESS_TOKEN"]}
    url = f"https://www.onemap.gov.sg/api/common/elastic/search?searchVal={address}&returnGeom=Y&getAddrDetails=Y"
    async with aiohttp.ClientSession(headers=headers) as session:
        response = await session.get(url)
        while response.status == 429:
            # Backing off for 10s
            time.sleep("10")
            response = await session.get(url)

        try:
            response_json = await response.json()
            result = response_json["results"][0]
            return (result["LATITUDE"], result["LONGITUDE"])
        except Exception as e:
            logger.warn(f"Request for {address}: error {str(e)}")
            logger.warn(f"Response {response_json}")
            return (None, None)


street_name_coords = {}


async def process_get_coordinates(row_id: int, row):
    if row["street_name"] not in street_name_coords:
        lat, long = await async_get_coordinates(row["street_name"])
        street_name_coords[row["street_name"]] = (lat, long)
    lat, long = street_name_coords[row["street_name"]]
    return row_id, lat, long


async def limit_openmapapi_calls(df: pd.DataFrame, func):
    df["house_lat"] = None
    df["house_long"] = None
    processes = [func(index, row) for index, row in df.iterrows()]
    it = iter(processes)
    batch_count = 0
    while True:
        print(f"batch count {batch_count}")
        batch = list(itertools.islice(it, 500))
        if not batch:
            break
        for result in await asyncio.gather(*batch):
            row_id, lat, long = result
            df.loc[row_id, "house_lat"] = lat
            df.loc[row_id, "house_long"] = long
        time.sleep(5)
        batch_count += 1

def check_for_new_data(prev_run_date: datetime, current_run_date: datetime) -> List[str]:
    url = "https://data.gov.sg/api/action/package_show?id=resale-flat-prices"
    response = request("GET", url)
    resources = response.json()["result"]["resources"]
    new_or_updated_resources = []
    for resource in resources:
        coverage_end = datetime.strptime(resource["coverage_end"], "%Y-%m-%d") 
        if coverage_end > prev_run_date and coverage_end < current_run_date:
            new_or_updated_resources.append(resource["url"])
    return new_or_updated_resources


def prepare_new_data(csv_urls: List[str]):
    dfs = []
    for csv_url in csv_urls:
        r = requests.get(csv_url)
        data = r.content.decode("utf8")
        raw_df = pd.read_csv(io.StringIO(data))
        if "remaining_lease" not in raw_df.columns:
            raw_df["remaining_lease"] = raw_df.apply(lambda row: relativedelta(datetime.strptime(row["month"], "%Y-%m"), 
                                                                               datetime.strptime(str(row["lease_commence_date"]), "%Y")).years, axis =1)
        dfs.append(raw_df)

    df = pd.concat(dfs, ignore_index=True)

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
    df["street_name"] = df.street_name.apply(lambda x: replace_address_abbreviation(x))
    logger.info(f"length of df: {len(df)}")

    logger.info(df.head())

    asyncio.run(limit_openmapapi_calls(df, process_get_coordinates))
    df = df.dropna(subset=["house_lat", "house_long"])

    # Get CBD travel distance
    df["distance_cbd"] = df.apply(lambda row: calculate_geodesic_dist(row["house_lat"], row["house_long"], RAFFLES_PLACE_LAT, RAFFLES_PLACE_LONG), axis=1)

    # Get distance to nearest popular primary school
    df["distance_top_primary"] = df.apply(lambda row: calculate_dist_to_nearest_primary_school(row["house_lat"], row["house_long"]), axis=1)

    # Get distance to nearest MRT
    df["distance_mrt"] = df.apply(lambda row: calculate_dist_to_nearest_mrt_station(row["house_lat"], row["house_long"]), axis=1)

    # Save csv
    df.to_csv(f"processed_{df['month'][0]}.csv")

    # Split the data into training and test sets. (0.75, 0.25) split.
    train, test = train_test_split(df)
    train, valid = train_test_split(train)
    columns_to_drop =[
            "block",
            "street_name",
            "lease_commence_date",
            "resale_price_per_sqm",
            "house_lat",
            "house_long",
            "month",
        ] 

    train_x = train.drop(
        columns=columns_to_drop
    )
    val_x = valid.drop(
        columns=columns_to_drop
    )
    test_x = test.drop(
        columns=columns_to_drop
    )
    train_y = train.resale_price_per_sqm.to_list()
    val_y = valid.resale_price_per_sqm.to_list()
    test_y = test.resale_price_per_sqm.to_list()
    return train_x, val_x, test_x, train_y, val_y, test_y
