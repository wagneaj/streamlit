import ee
from google.oauth2 import service_account
from ee import oauth
import geemap
import ipyleaflet
import folium
import pprint
import branca.colormap as cm
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from streamlit_folium import st_folium
import matplotlib.pyplot as plt
import datetime as datetime


st.title('California Water Workflow')

#Authenticate and Initialize Google Earth Engine
##GOOGLE_CLOUD_DISABLE_GRPC=True
key = st.secrets["credentials_key"]["key"]
service_account_info = st.secrets["credentials_sa"]["service_account"]

credentials = service_account.Credentials.from_service_account_info(key, scopes=oauth.SCOPES)
ee.Initialize(credentials)



# Initial date of interest (inclusive).
i_date = "2018-01-01"

# Final date of interest (exclusive).
f_date = "2023-01-01"

# Define the location of interest with a point.
lon = -119.7889
lat = 36.9915
poi = ee.Geometry.Point(lon, lat)

# A nominal scale in meters of the projection to work in [in meters].
scale = 1000

# Soil depths [in cm] where we have data.
olm_depths = [0, 10, 30, 60, 100, 200]

# Names of bands associated with reference depths.
olm_bands = ["b" + str(sd) for sd in olm_depths]


def get_soil_prop(param):
    """
    This function returns soil properties image
    param (str): must be one of:
        "sand"     - Sand fraction
        "clay"     - Clay fraction
        "orgc"     - Organic Carbon fraction
    """
    if param == "sand":  # Sand fraction [%w]
        snippet = "OpenLandMap/SOL/SOL_SAND-WFRACTION_USDA-3A1A1A_M/v02"
        # Define the scale factor in accordance with the dataset description.
        scale_factor = 1 * 0.01

    elif param == "clay":  # Clay fraction [%w]
        snippet = "OpenLandMap/SOL/SOL_CLAY-WFRACTION_USDA-3A1A1A_M/v02"
        # Define the scale factor in accordance with the dataset description.
        scale_factor = 1 * 0.01

    elif param == "orgc":  # Organic Carbon fraction [g/kg]
        snippet = "OpenLandMap/SOL/SOL_ORGANIC-CARBON_USDA-6A1C_M/v02"
        # Define the scale factor in accordance with the dataset description.
        scale_factor = 5 * 0.001  # to get kg/kg
    else:
        return print("error")

    # Apply the scale factor to the ee.Image.
    dataset = ee.Image(snippet).multiply(scale_factor)

    return dataset


# Image associated with the sand content.
sand = get_soil_prop("sand")

# Image associated with the clay content.
clay = get_soil_prop("clay")

# Image associated with the organic carbon content.
orgc = get_soil_prop("orgc")

def add_ee_layer(self, ee_image_object, vis_params, name):
    """Adds a method for displaying Earth Engine image tiles to folium map."""
    map_id_dict = ee.Image(ee_image_object).getMapId(vis_params)
    folium.raster_layers.TileLayer(
        tiles=map_id_dict["tile_fetcher"].url_format,
        attr="Map Data &copy; <a href='https://earthengine.google.com/'>Google Earth Engine</a>",
        name=name,
        overlay=True,
        control=True,
    ).add_to(self)





#Streamlit Columns
col1, col2 = st.columns(2)




# Get a feature collection of administrative boundaries.
countries = ee.FeatureCollection("TIGER/2016/States").select("NAME")

# Filter the feature collection to subset France.0
california = countries.filter(ee.Filter.eq("NAME", "California"))

# Clip the composite ee.Images around the region of interest.
sand_clip = sand.clip(california)
clay_clip = clay.clip(california)

# Add Earth Engine drawing method to folium.
folium.Map.add_ee_layer = add_ee_layer

hyd_map = folium.Map(location=[lat, lon], titles="Hydraulic Content", zoom_start=6)

# Set visualization parameters.
sand_vis_params = {
    "bands": ["b0"],
    "min": 0.01,
    "max": 1,
    "opacity": 1,
    "palette": ["white", "orange", "brown"],
}

clay_vis_params = {
    "bands": ["b0"],
    "min": 0.01,
    "max": 1,
    "opacity": 1,
    "palette": ["white", "pink", "red"],
}



# Add the sand content data to the map object.
hyd_map.add_ee_layer(sand_clip, sand_vis_params, "Sand Content")
hyd_map.add_ee_layer(clay_clip, clay_vis_params, "Clay Content")
#my_map.add_ee_layer(orgc, orgc_vis_params, "Organic Carbon Content")

# Add a marker at the location of interest.
#folium.Marker([lat, lon], popup="point of interest").add_to(my_map)

# Add a layer control panel to the map.
hyd_map.add_child(folium.LayerControl())

#st.header("California Hydraulic Properties")
    
# Display the map
#st_folium(my_map)



# Conversion of organic carbon content into organic matter content.
orgm = orgc.multiply(1.724)



# Initialization of two constant images for wilting point and field capacity.
wilting_point = ee.Image(0)
field_capacity = ee.Image(0)

# Calculation for each standard depth using a loop.
for key in olm_bands:
    # Getting sand, clay and organic matter at the appropriate depth.
    si = sand.select(key)
    ci = clay.select(key)
    oi = orgm.select(key)

    # Calculation of the wilting point.
    # The theta_1500t parameter is needed for the given depth.
    theta_1500ti = (
        ee.Image(0)
        .expression(
            "-0.024 * S + 0.487 * C + 0.006 * OM + 0.005 * (S * OM)\
        - 0.013 * (C * OM) + 0.068 * (S * C) + 0.031",
            {
                "S": si,
                "C": ci,
                "OM": oi,
            },
        )
        .rename("T1500ti")
    )

    # Final expression for the wilting point.
    wpi = theta_1500ti.expression(
        "T1500ti + ( 0.14 * T1500ti - 0.002)", {"T1500ti": theta_1500ti}
    ).rename("wpi")

    # Add as a new band of the global wilting point ee.Image.
    # Do not forget to cast the type with float().
    wilting_point = wilting_point.addBands(wpi.rename(key).float())

    # Same process for the calculation of the field capacity.
    # The parameter theta_33t is needed for the given depth.
    theta_33ti = (
        ee.Image(0)
        .expression(
            "-0.251 * S + 0.195 * C + 0.011 * OM +\
        0.006 * (S * OM) - 0.027 * (C * OM)+\
        0.452 * (S * C) + 0.299",
            {
                "S": si,
                "C": ci,
                "OM": oi,
            },
        )
        .rename("T33ti")
    )

    # Final expression for the field capacity of the soil.
    fci = theta_33ti.expression(
        "T33ti + (1.283 * T33ti * T33ti - 0.374 * T33ti - 0.015)",
        {"T33ti": theta_33ti.select("T33ti")},
    )

    # Add a new band of the global field capacity ee.Image.
    field_capacity = field_capacity.addBands(fci.rename(key).float())




# Import precipitation.
pr = (
    ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")
    .select("precipitation")
    .filterDate(i_date, f_date)
)

# Import potential evaporation PET and its quality indicator ET_QC.
pet = (
    ee.ImageCollection("MODIS/006/MOD16A2")
    .select(["PET", "ET_QC"])
    .filterDate(i_date, f_date)
)





def sum_resampler(coll, freq, unit, scale_factor, band_name):
    """
    This function aims to resample the time scale of an ee.ImageCollection.
    The function returns an ee.ImageCollection with the averaged sum of the
    band on the selected frequency.
    coll: (ee.ImageCollection) only one band can be handled
    freq: (int) corresponds to the resampling frequence
    unit: (str) corresponds to the resampling time unit.
                must be 'day', 'month' or 'year'
    scale_factor (float): scaling factor used to get our value in the good unit
    band_name (str) name of the output band
    """

    # Define initial and final dates of the collection.
    firstdate = ee.Date(
        coll.sort("system:time_start", True).first().get("system:time_start")
    )

    lastdate = ee.Date(
        coll.sort("system:time_start", False).first().get("system:time_start")
    )

    # Calculate the time difference between both dates.
    # https://developers.google.com/earth-engine/apidocs/ee-date-difference
    diff_dates = lastdate.difference(firstdate, unit)

    # Define a new time index (for output).
    new_index = ee.List.sequence(0, ee.Number(diff_dates), freq)

    # Define the function that will be applied to our new time index.
    def apply_resampling(date_index):
        # Define the starting date to take into account.
        startdate = firstdate.advance(ee.Number(date_index), unit)

        # Define the ending date to take into account according
        # to the desired frequency.
        enddate = firstdate.advance(ee.Number(date_index).add(freq), unit)

        # Calculate the number of days between starting and ending days.
        diff_days = enddate.difference(startdate, "day")

        # Calculate the composite image.
        image = (
            coll.filterDate(startdate, enddate)
            .mean()
            .multiply(diff_days)
            .multiply(scale_factor)
            .rename(band_name)
        )

        # Return the final image with the appropriate time index.
        return image.set("system:time_start", startdate.millis())

    # Map the function to the new time index.
    res = new_index.map(apply_resampling)

    # Transform the result into an ee.ImageCollection.
    res = ee.ImageCollection(res)

    return res




# Apply the resampling function to the precipitation dataset.
pr_m = sum_resampler(pr, 1, "month", 1, "pr")


# Apply the resampling function to the PET dataset.
pet_m = sum_resampler(pet.select("PET"), 1, "month", 0.0125, "pet")

# Combine precipitation and evapotranspiration.
meteo = pr_m.combine(pet_m)


zr = ee.Image(0.5)
p = ee.Image(0.5)

def olm_prop_mean(olm_image, band_output_name):
    """
    This function calculates an averaged value of
    soil properties between reference depths.
    """
    mean_image = olm_image.expression(
        "(b0 + b10 + b30 + b60 + b100 + b200) / 6",
        {
            "b0": olm_image.select("b0"),
            "b10": olm_image.select("b10"),
            "b30": olm_image.select("b30"),
            "b60": olm_image.select("b60"),
            "b100": olm_image.select("b100"),
            "b200": olm_image.select("b200"),
        },
    ).rename(band_output_name)

    return mean_image


# Apply the function to field capacity and wilting point.
fcm = olm_prop_mean(field_capacity, "fc_mean")
wpm = olm_prop_mean(wilting_point, "wp_mean")

# Calculate the theoretical available water.
taw = (
    (fcm.select("fc_mean").subtract(wpm.select("wp_mean"))).multiply(1000).multiply(zr)
)

# Calculate the stored water at the field capacity.
stfc = taw.multiply(p)

# Define the initial time (time0) according to the start of the collection.
time0 = meteo.first().get("system:time_start")

# Initialize all bands describing the hydric state of the soil.
# Do not forget to cast the type of the data with a .float().
# Initial recharge.
initial_rech = ee.Image(0).set("system:time_start", time0).select([0], ["rech"]).float()

# Initialization of APWL.
initial_apwl = ee.Image(0).set("system:time_start", time0).select([0], ["apwl"]).float()

# Initialization of ST.
initial_st = stfc.set("system:time_start", time0).select([0], ["st"]).float()

# Initialization of precipitation.
initial_pr = ee.Image(0).set("system:time_start", time0).select([0], ["pr"]).float()

# Initialization of potential evapotranspiration.
initial_pet = ee.Image(0).set("system:time_start", time0).select([0], ["pet"]).float()

initial_image = initial_rech.addBands(
    ee.Image([initial_apwl, initial_st, initial_pr, initial_pet])
)




image_list = ee.List([initial_image])





def recharge_calculator(image, image_list):
    """
    Contains operations made at each iteration.
    """
    # Determine the date of the current ee.Image of the collection.
    localdate = image.date().millis()

    # Import previous image stored in the list.
    prev_im = ee.Image(ee.List(image_list).get(-1))

    # Import previous APWL and ST.
    prev_apwl = prev_im.select("apwl")
    prev_st = prev_im.select("st")

    # Import current precipitation and evapotranspiration.
    pr_im = image.select("pr").add(1)
    pet_im = image.select("pet")

    # Initialize the new bands associated with recharge, apwl and st.
    # DO NOT FORGET TO CAST THE TYPE WITH .float().
    new_rech = (
        ee.Image(0)
        .set("system:time_start", localdate)
        .select([0], ["rech"])
        .float()
    )

    new_apwl = (
        ee.Image(0)
        .set("system:time_start", localdate)
        .select([0], ["apwl"])
        .float()
    )

    new_st = (
        prev_st.set("system:time_start", localdate).select([0], ["st"]).float()
    )

    # Calculate bands depending on the situation using binary layers with
    # logical operations.

    # CASE 1.
    # Define zone1: the area where PET > P.
    zone1 = pet_im.gt(pr_im)

    # Calculation of APWL in zone 1.
    zone1_apwl = prev_apwl.add(pet_im.subtract(pr_im)).rename("apwl")
    # Implementation of zone 1 values for APWL.
    new_apwl = new_apwl.where(zone1, zone1_apwl)

    # Calculate ST in zone 1.
    zone1_st = prev_st.multiply(
        ee.Image.exp(zone1_apwl.divide(stfc).multiply(-1))
    ).rename("st")
    # Implement ST in zone 1.
    new_st = new_st.where(zone1, zone1_st)

    # CASE 2.
    # Define zone2: the area where PET <= P.
    zone2 = pet_im.lte(pr_im)

    # Calculate ST in zone 2.
    zone2_st = prev_st.add(pr_im).subtract(pet_im).rename("st")
    # Implement ST in zone 2.
    new_st = new_st.where(zone2, zone2_st)

    # CASE 2.1.
    # Define zone21: the area where PET <= P and ST >= STfc.
    zone21 = zone2.And(zone2_st.gte(stfc))

    # Calculate recharge in zone 21.
    zone21_re = zone2_st.subtract(stfc).rename("rech")
    # Implement recharge in zone 21.
    new_rech = new_rech.where(zone21, zone21_re)
    # Implement ST in zone 21.
    new_st = new_st.where(zone21, stfc)

    # CASE 2.2.
    # Define zone 22: the area where PET <= P and ST < STfc.
    zone22 = zone2.And(zone2_st.lt(stfc))

    # Calculate APWL in zone 22.
    zone22_apwl = (
        stfc.multiply(-1).multiply(ee.Image.log(zone2_st.divide(stfc))).rename("apwl")
    )
    # Implement APWL in zone 22.
    new_apwl = new_apwl.where(zone22, zone22_apwl)

    # Create a mask around area where recharge can effectively be calculated.
    # Where we have have PET, P, FCm, WPm (except urban areas, etc.).
    mask = pet_im.gte(0).And(pr_im.gte(0)).And(fcm.gte(0)).And(wpm.gte(0))

    # Apply the mask.
    new_rech = new_rech.updateMask(mask)

    # Add all Bands to our ee.Image.
    new_image = new_rech.addBands(ee.Image([new_apwl, new_st, pr_im, pet_im]))

    # Add the new ee.Image to the ee.List.
    return ee.List(image_list).add(new_image)




# Iterate the user-supplied function to the meteo collection.
rech_list = meteo.iterate(recharge_calculator, image_list)

# Remove the initial image from our list.
rech_list = ee.List(rech_list).remove(initial_image)

# Transform the list into an ee.ImageCollection.
rech_coll = ee.ImageCollection(rech_list)

# Calculate the averaged annual recharge.
annual_rech = rech_coll.select("rech").mean().multiply(12)

# Calculate the average annual precipitation.
annual_pr = rech_coll.select("pr").mean().multiply(12)

# Get a feature collection of administrative boundaries.
countries = ee.FeatureCollection("TIGER/2016/States").select("NAME")


counties = ee.FeatureCollection("TIGER/2016/Counties").select("NAME")


# Filter the feature collection to subset France.0
france = countries.filter(ee.Filter.eq("NAME", "California"))

central_valley = counties.filter(ee.Filter.eq("NAME", ["Butte", "Colusa", "Glenn", "Fresno", "Kern", "Kings", "Madera", "Merced", "Placer", "San Joaquin","Sacramento", "Shasta", "Solano", "Stanislaus", "Sutter", "Tehama", "Tulare", "Yolo", "Yuba"]))

# Clip the composite ee.Images around the region of interest.
rech_france = annual_rech.clip(france)
pr_france = annual_pr.clip(france)

####################################################

def recharge_calculator_20(image, image_list):
    """
    Contains operations made at each iteration.
    """
    # Determine the date of the current ee.Image of the collection.
    localdate = image.date().millis()

    # Import previous image stored in the list.
    prev_im = ee.Image(ee.List(image_list).get(-1))

    # Import previous APWL and ST.
    prev_apwl = prev_im.select("apwl")
    prev_st = prev_im.select("st")

    # Import current precipitation and evapotranspiration.
    pr_im = image.select("pr").add(20)
    pet_im = image.select("pet")

    # Initialize the new bands associated with recharge, apwl and st.
    # DO NOT FORGET TO CAST THE TYPE WITH .float().
    new_rech = (
        ee.Image(0)
        .set("system:time_start", localdate)
        .select([0], ["rech"])
        .float()
    )

    new_apwl = (
        ee.Image(0)
        .set("system:time_start", localdate)
        .select([0], ["apwl"])
        .float()
    )

    new_st = (
        prev_st.set("system:time_start", localdate).select([0], ["st"]).float()
    )

    # Calculate bands depending on the situation using binary layers with
    # logical operations.

    # CASE 1.
    # Define zone1: the area where PET > P.
    zone1 = pet_im.gt(pr_im)

    # Calculation of APWL in zone 1.
    zone1_apwl = prev_apwl.add(pet_im.subtract(pr_im)).rename("apwl")
    # Implementation of zone 1 values for APWL.
    new_apwl = new_apwl.where(zone1, zone1_apwl)

    # Calculate ST in zone 1.
    zone1_st = prev_st.multiply(
        ee.Image.exp(zone1_apwl.divide(stfc).multiply(-1))
    ).rename("st")
    # Implement ST in zone 1.
    new_st = new_st.where(zone1, zone1_st)

    # CASE 2.
    # Define zone2: the area where PET <= P.
    zone2 = pet_im.lte(pr_im)

    # Calculate ST in zone 2.
    zone2_st = prev_st.add(pr_im).subtract(pet_im).rename("st")
    # Implement ST in zone 2.
    new_st = new_st.where(zone2, zone2_st)

    # CASE 2.1.
    # Define zone21: the area where PET <= P and ST >= STfc.
    zone21 = zone2.And(zone2_st.gte(stfc))

    # Calculate recharge in zone 21.
    zone21_re = zone2_st.subtract(stfc).rename("rech")
    # Implement recharge in zone 21.
    new_rech = new_rech.where(zone21, zone21_re)
    # Implement ST in zone 21.
    new_st = new_st.where(zone21, stfc)

    # CASE 2.2.
    # Define zone 22: the area where PET <= P and ST < STfc.
    zone22 = zone2.And(zone2_st.lt(stfc))

    # Calculate APWL in zone 22.
    zone22_apwl = (
        stfc.multiply(-1).multiply(ee.Image.log(zone2_st.divide(stfc))).rename("apwl")
    )
    # Implement APWL in zone 22.
    new_apwl = new_apwl.where(zone22, zone22_apwl)

    # Create a mask around area where recharge can effectively be calculated.
    # Where we have have PET, P, FCm, WPm (except urban areas, etc.).
    mask = pet_im.gte(0).And(pr_im.gte(0)).And(fcm.gte(0)).And(wpm.gte(0))

    # Apply the mask.
    new_rech = new_rech.updateMask(mask)

    # Add all Bands to our ee.Image.
    new_image = new_rech.addBands(ee.Image([new_apwl, new_st, pr_im, pet_im]))

    # Add the new ee.Image to the ee.List.
    return ee.List(image_list).add(new_image)




# Iterate the user-supplied function to the meteo collection.
rech_list_20 = meteo.iterate(recharge_calculator_20, image_list)

# Remove the initial image from our list.
rech_list_20 = ee.List(rech_list_20).remove(initial_image)

# Transform the list into an ee.ImageCollection.
rech_coll_20 = ee.ImageCollection(rech_list_20)

# Calculate the averaged annual recharge.
annual_rech_20 = rech_coll_20.select("rech").mean().multiply(12)

# Calculate the average annual precipitation.
annual_pr_20 = rech_coll_20.select("pr").mean().multiply(12)

# Get a feature collection of administrative boundaries.
countries = ee.FeatureCollection("TIGER/2016/States").select("NAME")

# Filter the feature collection to subset France.0
france = countries.filter(ee.Filter.eq("NAME", "California"))

# Clip the composite ee.Images around the region of interest.
rech_france_20 = annual_rech_20.clip(france)
pr_france_20 = annual_pr_20.clip(france)

#########################################

def recharge_calculator_40(image, image_list):
    """
    Contains operations made at each iteration.
    """
    # Determine the date of the current ee.Image of the collection.
    localdate = image.date().millis()

    # Import previous image stored in the list.
    prev_im = ee.Image(ee.List(image_list).get(-1))

    # Import previous APWL and ST.
    prev_apwl = prev_im.select("apwl")
    prev_st = prev_im.select("st")

    # Import current precipitation and evapotranspiration.
    pr_im = image.select("pr").add(40)
    pet_im = image.select("pet")

    # Initialize the new bands associated with recharge, apwl and st.
    # DO NOT FORGET TO CAST THE TYPE WITH .float().
    new_rech = (
        ee.Image(0)
        .set("system:time_start", localdate)
        .select([0], ["rech"])
        .float()
    )

    new_apwl = (
        ee.Image(0)
        .set("system:time_start", localdate)
        .select([0], ["apwl"])
        .float()
    )

    new_st = (
        prev_st.set("system:time_start", localdate).select([0], ["st"]).float()
    )

    # Calculate bands depending on the situation using binary layers with
    # logical operations.

    # CASE 1.
    # Define zone1: the area where PET > P.
    zone1 = pet_im.gt(pr_im)

    # Calculation of APWL in zone 1.
    zone1_apwl = prev_apwl.add(pet_im.subtract(pr_im)).rename("apwl")
    # Implementation of zone 1 values for APWL.
    new_apwl = new_apwl.where(zone1, zone1_apwl)

    # Calculate ST in zone 1.
    zone1_st = prev_st.multiply(
        ee.Image.exp(zone1_apwl.divide(stfc).multiply(-1))
    ).rename("st")
    # Implement ST in zone 1.
    new_st = new_st.where(zone1, zone1_st)

    # CASE 2.
    # Define zone2: the area where PET <= P.
    zone2 = pet_im.lte(pr_im)

    # Calculate ST in zone 2.
    zone2_st = prev_st.add(pr_im).subtract(pet_im).rename("st")
    # Implement ST in zone 2.
    new_st = new_st.where(zone2, zone2_st)

    # CASE 2.1.
    # Define zone21: the area where PET <= P and ST >= STfc.
    zone21 = zone2.And(zone2_st.gte(stfc))

    # Calculate recharge in zone 21.
    zone21_re = zone2_st.subtract(stfc).rename("rech")
    # Implement recharge in zone 21.
    new_rech = new_rech.where(zone21, zone21_re)
    # Implement ST in zone 21.
    new_st = new_st.where(zone21, stfc)

    # CASE 2.2.
    # Define zone 22: the area where PET <= P and ST < STfc.
    zone22 = zone2.And(zone2_st.lt(stfc))

    # Calculate APWL in zone 22.
    zone22_apwl = (
        stfc.multiply(-1).multiply(ee.Image.log(zone2_st.divide(stfc))).rename("apwl")
    )
    # Implement APWL in zone 22.
    new_apwl = new_apwl.where(zone22, zone22_apwl)

    # Create a mask around area where recharge can effectively be calculated.
    # Where we have have PET, P, FCm, WPm (except urban areas, etc.).
    mask = pet_im.gte(0).And(pr_im.gte(0)).And(fcm.gte(0)).And(wpm.gte(0))

    # Apply the mask.
    new_rech = new_rech.updateMask(mask)

    # Add all Bands to our ee.Image.
    new_image = new_rech.addBands(ee.Image([new_apwl, new_st, pr_im, pet_im]))

    # Add the new ee.Image to the ee.List.
    return ee.List(image_list).add(new_image)




# Iterate the user-supplied function to the meteo collection.
rech_list_40 = meteo.iterate(recharge_calculator_40, image_list)

# Remove the initial image from our list.
rech_list_40 = ee.List(rech_list_40).remove(initial_image)

# Transform the list into an ee.ImageCollection.
rech_coll_40 = ee.ImageCollection(rech_list_40)

# Calculate the averaged annual recharge.
annual_rech_40 = rech_coll_40.select("rech").mean().multiply(12)

# Calculate the average annual precipitation.
annual_pr_40 = rech_coll_40.select("pr").mean().multiply(12)

# Get a feature collection of administrative boundaries.
countries = ee.FeatureCollection("TIGER/2016/States").select("NAME")

# Filter the feature collection to subset France.0
france = countries.filter(ee.Filter.eq("NAME", "California"))

# Clip the composite ee.Images around the region of interest.
rech_france_40 = annual_rech_40.clip(france)
pr_france_40 = annual_pr_40.clip(france)

#####################################

def recharge_calculator_60(image, image_list):
    """
    Contains operations made at each iteration.
    """
    # Determine the date of the current ee.Image of the collection.
    localdate = image.date().millis()

    # Import previous image stored in the list.
    prev_im = ee.Image(ee.List(image_list).get(-1))

    # Import previous APWL and ST.
    prev_apwl = prev_im.select("apwl")
    prev_st = prev_im.select("st")

    # Import current precipitation and evapotranspiration.
    pr_im = image.select("pr").add(60)
    pet_im = image.select("pet")

    # Initialize the new bands associated with recharge, apwl and st.
    # DO NOT FORGET TO CAST THE TYPE WITH .float().
    new_rech = (
        ee.Image(0)
        .set("system:time_start", localdate)
        .select([0], ["rech"])
        .float()
    )

    new_apwl = (
        ee.Image(0)
        .set("system:time_start", localdate)
        .select([0], ["apwl"])
        .float()
    )

    new_st = (
        prev_st.set("system:time_start", localdate).select([0], ["st"]).float()
    )

    # Calculate bands depending on the situation using binary layers with
    # logical operations.

    # CASE 1.
    # Define zone1: the area where PET > P.
    zone1 = pet_im.gt(pr_im)

    # Calculation of APWL in zone 1.
    zone1_apwl = prev_apwl.add(pet_im.subtract(pr_im)).rename("apwl")
    # Implementation of zone 1 values for APWL.
    new_apwl = new_apwl.where(zone1, zone1_apwl)

    # Calculate ST in zone 1.
    zone1_st = prev_st.multiply(
        ee.Image.exp(zone1_apwl.divide(stfc).multiply(-1))
    ).rename("st")
    # Implement ST in zone 1.
    new_st = new_st.where(zone1, zone1_st)

    # CASE 2.
    # Define zone2: the area where PET <= P.
    zone2 = pet_im.lte(pr_im)

    # Calculate ST in zone 2.
    zone2_st = prev_st.add(pr_im).subtract(pet_im).rename("st")
    # Implement ST in zone 2.
    new_st = new_st.where(zone2, zone2_st)

    # CASE 2.1.
    # Define zone21: the area where PET <= P and ST >= STfc.
    zone21 = zone2.And(zone2_st.gte(stfc))

    # Calculate recharge in zone 21.
    zone21_re = zone2_st.subtract(stfc).rename("rech")
    # Implement recharge in zone 21.
    new_rech = new_rech.where(zone21, zone21_re)
    # Implement ST in zone 21.
    new_st = new_st.where(zone21, stfc)

    # CASE 2.2.
    # Define zone 22: the area where PET <= P and ST < STfc.
    zone22 = zone2.And(zone2_st.lt(stfc))

    # Calculate APWL in zone 22.
    zone22_apwl = (
        stfc.multiply(-1).multiply(ee.Image.log(zone2_st.divide(stfc))).rename("apwl")
    )
    # Implement APWL in zone 22.
    new_apwl = new_apwl.where(zone22, zone22_apwl)

    # Create a mask around area where recharge can effectively be calculated.
    # Where we have have PET, P, FCm, WPm (except urban areas, etc.).
    mask = pet_im.gte(0).And(pr_im.gte(0)).And(fcm.gte(0)).And(wpm.gte(0))

    # Apply the mask.
    new_rech = new_rech.updateMask(mask)

    # Add all Bands to our ee.Image.
    new_image = new_rech.addBands(ee.Image([new_apwl, new_st, pr_im, pet_im]))

    # Add the new ee.Image to the ee.List.
    return ee.List(image_list).add(new_image)




# Iterate the user-supplied function to the meteo collection.
rech_list_60 = meteo.iterate(recharge_calculator_60, image_list)

# Remove the initial image from our list.
rech_list_60 = ee.List(rech_list_60).remove(initial_image)

# Transform the list into an ee.ImageCollection.
rech_coll_60 = ee.ImageCollection(rech_list_60)

# Calculate the averaged annual recharge.
annual_rech_60 = rech_coll_60.select("rech").mean().multiply(12)

# Calculate the average annual precipitation.
annual_pr_60 = rech_coll_60.select("pr").mean().multiply(12)

# Get a feature collection of administrative boundaries.
countries = ee.FeatureCollection("TIGER/2016/States").select("NAME")

# Filter the feature collection to subset France.0
france = countries.filter(ee.Filter.eq("NAME", "California"))

# Clip the composite ee.Images around the region of interest.
rech_france_60 = annual_rech_60.clip(france)
pr_france_60 = annual_pr_60.clip(france)

#########################################################



# Create a folium map for precipitation and evapotranspiration.
rech_map = folium.Map(location=[lat, lon], zoom_start=6, width=10000, heigth=10000, zoom_control=False)

# Set visualization parameters for recharge.
rech_vis_params = {
    "bands": "rech",
    "min": 0,
    "max": 500,
    "opacity": 1,
    "palette": ["red", "orange", "yellow", "green", "blue", "purple"],
}

# Set visualization parameters for precipitation.
pr_vis_params = {
    "bands": "pr",
    "min": 500,
    "max": 1500,
    "opacity": 1,
    "palette": ["white", "blue"],
}


#Set visualization parameters for Central Valley polygon
cv_vis_params = {
    "color":'black'
}

# Define a recharge colormap.
rech_colormap = cm.LinearColormap(
    colors=rech_vis_params["palette"],
    vmin=rech_vis_params["min"],
    vmax=rech_vis_params["max"],
)

# Define a precipitation colormap.
pr_colormap = cm.LinearColormap(
    colors=pr_vis_params["palette"],
    vmin=pr_vis_params["min"],
    vmax=pr_vis_params["max"],
)

# Caption of the recharge colormap.
rech_colormap.caption = "Average annual recharge rate (mm/year)"

# Caption of the precipitation colormap.
pr_colormap.caption = "Average annual precipitation rate (mm/year)"

# Add the precipitation composite to the map object.
rech_map.add_ee_layer(pr_france, pr_vis_params, "Precipitation")

# Add the recharge composite to the map object.
rech_map.add_ee_layer(rech_france, rech_vis_params, "Recharge")

# Add the recharge20 composite to the map object.
rech_map.add_ee_layer(rech_france_20, rech_vis_params, "Recharge20")

# Add the recharge40 composite to the map object.
rech_map.add_ee_layer(rech_france_40, rech_vis_params, "Recharge40")

# Add the recharge60 composite to the map object.
rech_map.add_ee_layer(rech_france_60, rech_vis_params, "Recharge60")

#my_map.add_ee_layer(central_valley, cv_vis_params, "Central Valley" )

# Add a marker at both locations of interest.
##folium.Marker([lat, lon], popup="Area of Lyon").add_to(rech_map)


# Add the colormaps to the map.
rech_map.add_child(rech_colormap)
rech_map.add_child(pr_colormap)

# Add a layer control panel to the map.
rech_map.add_child(folium.LayerControl())



#st.header("California Ground Water Recharge")
# Display the map.
#st_folium(my_map)




#Sidebar

st.sidebar.subheader("Select a Map")
add_radio = st.sidebar.radio( 
    "Map View",
    ("Hydraulic Content", "Groundwater Recharge Rates", "Data Connections")
)

if add_radio == "Hydraulic Content":
    st.subheader("California Soil Hydric Properties")
    st_folium(hyd_map)

elif add_radio == 'Groundwater Recharge Rates':
    st.subheader("California Groundwater Recharge")
    st_folium(rech_map)

elif add_radio == "Data Connections":
    st.subheader("California Water Workflow Data Connections")
    video_file = open(r"Downloads\data_connections", 'rb')
    video_bytes = video_file.read()

    st.video(video_bytes)
    


st.sidebar.text(" ")
st.sidebar.text(" ")
st.sidebar.text(" ")
st.sidebar.text(" ")
st.sidebar.text(" ")
st.sidebar.text(" ")
st.sidebar.text(" ")
st.sidebar.text(" ")
st.sidebar.text(" ")
st.sidebar.text(" ")
st.sidebar.text(" ")
st.sidebar.text(" ")
st.sidebar.text(" ")
st.sidebar.text(" ")
st.sidebar.text(" ")
st.sidebar.text(" ")
st.sidebar.text(" ")
st.sidebar.text(" ")
st.sidebar.text(" ")
st.sidebar.text(" ")
st.sidebar.text(" ")
st.sidebar.text(" ")
st.sidebar.text(" ")
st.sidebar.text(" ")
st.sidebar.text(" ")
st.sidebar.text(" ")
st.sidebar.text(" ")


image = r"Downloads\seerai_alt_glow"
st.sidebar.image(image, use_column_width=True)