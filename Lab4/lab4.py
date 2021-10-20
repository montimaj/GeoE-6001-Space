import ee
import pandas as pd
import matplotlib.pyplot as plt
import requests
import geopandas as gpd
from tqdm import tqdm
import zipfile
import os
import geemap

# initialize google earth engine if it hasn't been initialized yet
if not ee.data._credentials:
    # ee.Authenticate()
    ee.Initialize()

# functions for use in the lab. ---------------------------
# some of these are used in the base code, some are not but are included
# for those who want to explore the data further

def create_reduce_region_function(geometry,
                                  reducer=ee.Reducer.mean(),
                                  scale=1000,
                                  crs='EPSG:4326',
                                  bestEffort=True,
                                  maxPixels=1e13,
                                  tileScale=4):
  """Creates a region reduction function.

  Creates a region reduction function intended to be used as the input function
  to ee.ImageCollection.map() for reducing pixels intersecting a provided region
  to a statistic for each image in a collection. See ee.Image.reduceRegion()
  documentation for more details.

  Args:
    geometry:
      An ee.Geometry that defines the region over which to reduce data.
    reducer:
      Optional; An ee.Reducer that defines the reduction method.
    scale:
      Optional; A number that defines the nominal scale in meters of the
      projection to work in.
    crs:
      Optional; An ee.Projection or EPSG string ('EPSG:5070') that defines
      the projection to work in.
    bestEffort:
      Optional; A Boolean indicator for whether to use a larger scale if the
      geometry contains too many pixels at the given scale for the operation
      to succeed.
    maxPixels:
      Optional; A number specifying the maximum number of pixels to reduce.
    tileScale:
      Optional; A number representing the scaling factor used to reduce
      aggregation tile size; using a larger tileScale (e.g. 2 or 4) may enable
      computations that run out of memory with the default.

  Returns:
    A function that accepts an ee.Image and reduces it by region, according to
    the provided arguments.
  """

  def reduce_region_function(img):
    """Applies the ee.Image.reduceRegion() method.

    Args:
      img:
        An ee.Image to reduce to a statistic by region.

    Returns:
      An ee.Feature that contains properties representing the image region
      reduction results per band and the image timestamp formatted as
      milliseconds from Unix epoch (included to enable time series plotting).
    """

    stat = img.reduceRegion(
        reducer=reducer,
        geometry=geometry,
        scale=scale,
        crs=crs,
        bestEffort=bestEffort,
        maxPixels=maxPixels,
        tileScale=tileScale)

    return ee.Feature(geometry, stat).set({'millis': img.date().millis()})
  return reduce_region_function


def gee_zonal_mean_img_coll(imageCollection,geometry):
    reduce_iC = create_reduce_region_function(geometry=geometry)
    stat_fc = ee.FeatureCollection(imageCollection.map(reduce_iC)).filter(ee.Filter.notNull(imageCollection.first().bandNames()))
    fc_dict = fc_to_dict(stat_fc).getInfo()

    df = pd.DataFrame(fc_dict)
    df['date'] = pd.to_datetime(df['millis'], unit='ms')
    return(df)


def gee_zonal_mean(date1,date2,geometry,collection_name,band_name):
     imcol = ee.ImageCollection(collection_name).select(band_name).filterDate(date1,date2)
     df = gee_zonal_mean_img_coll(imcol,geometry)
     return(df)


# Define a function to transfer feature properties to a dictionary.
def fc_to_dict(fc):
  prop_names = fc.first().propertyNames()
  prop_lists = fc.reduceColumns(
      reducer=ee.Reducer.toList().repeat(prop_names.size()),
      selectors=prop_names).get('list')

  return ee.Dictionary.fromLists(prop_names, prop_lists)


def ee_imgcoll_to_df_point(imagecollection, lat,lon):
    """Transforms client-side ee.Image.getRegion array to pandas.DataFrame."""
    poi = ee.Geometry.Point(lon, lat)
    arr = imagecollection.getRegion(poi,1000).getInfo()

    list_of_bands = imagecollection.first().bandNames().getInfo()

    df = pd.DataFrame(arr)

    # Rearrange the header.
    headers = df.iloc[0]
    df = pd.DataFrame(df.values[1:], columns=headers)

    # Remove rows without data inside.
    df = df[['longitude', 'latitude', 'time', *list_of_bands]].dropna()

    # Convert the data to numeric values.
    for band in list_of_bands:
        df[band] = pd.to_numeric(df[band], errors='coerce')

    # Convert the time field into a datetime.
    df['datetime'] = pd.to_datetime(df['time'], unit='ms')

    # Keep the columns of interest.
    df = df[['time','datetime',  *list_of_bands]]

    return df

# to get the link to download an earth engine image
def getLink(image,fname,aoi):
  link = image.getDownloadURL({
    'scale': 1000,
    'crs': 'EPSG:4326',
    'fileFormat': 'GeoTIFF',
    'region': aoi,
    'name': fname})
  # print(link)
  return(link)
  
# create an earth engine geometry polygon
def addGeometry(min_lon,max_lon,min_lat,max_lat):
  geom = ee.Geometry.Polygon(
      [[[min_lon, max_lat],
        [min_lon, min_lat],
        [max_lon, min_lat],
        [max_lon, max_lat]]])
  return(geom)

def get_imgcollection(date1,date2,geometry,collection_name,band_name,function='mean'):
  collection = ee.ImageCollection(collection_name)
  if function=='mean':
      img = collection.filterDate(date1,date2).select(band_name).mean().clip(geometry)
  if function=='sum':
      img = collection.filterDate(date1,date2).select(band_name).sum().clip(geometry)
  range = img.reduceRegion(ee.Reducer.percentile([1, 99]),scale=10000)
  vals = range.getInfo()
  min=list(vals.items())[0][1]
  max=list(vals.items())[1][1]
  visParams = {'min': min, 'max': max, 'palette': ['0000FF', 'FFFFFF','FF0000']}
  return(img,visParams)    

def download_img(img,geom,fname):
    linkname = getLink(img,fname,geom)
    response = requests.get(linkname, stream=True)
    zipped = fname+'.zip'
    with open(zipped, "wb") as handle:
        for data in tqdm(response.iter_content()):
            handle.write(data)
    
    with zipfile.ZipFile(zipped, 'r') as zip_ref:
        zip_ref.extractall('')
    os.remove(zipped)
          

def aggregate_by_water_year(df,date_col,agg_column,agg_fun='sum'):
    df['water_year'] = df[date_col].dt.year.where(df[date_col].dt.month < 10, df[date_col].dt.year + 1)
    df_agg = df.groupby('water_year').agg({agg_column:[agg_fun]})
    return(df_agg)
    
plt.close('all')


# replace these with your own data--------------------------
start = '2015-01-01'
end = '2021-08-01'
path_to_watershed = 'Watershed/Salt_River.shp'
path_to_save_map = 'Figures/watershed_analysis.html'


# upload watershed to GEE----------------------------
watershed_gee = geemap.shp_to_ee(path_to_watershed)
gee_aoi = watershed_gee.geometry()

# get data frames that are spatially reduced-----------------------
gpm_df = gee_zonal_mean(start, end, gee_aoi, 'NASA/GPM_L3/IMERG_MONTHLY_V06', 'precipitation')
gpm_df['precipitation'] = gpm_df['precipitation'] * 24 * 365 / 12  # convert to monthly precip
gpm_agg = aggregate_by_water_year(gpm_df, 'date', 'precipitation')  # get water year aggregate

soil_moisture = gee_zonal_mean(start, end, gee_aoi, 'NASA_USDA/HSL/SMAP10KM_soil_moisture', 'ssm')
soil_moisture.ssm = soil_moisture.ssm * 365 / 3 / 12

mod16_df = gee_zonal_mean(start, end, gee_aoi, 'MODIS/006/MOD16A2', 'ET')
mod16_df.ET = mod16_df.ET * 365 / 8 / 12 / 10  # convert from every 8 days to monthly ET, divide by factor of 10

# get google earth engine rasters that are temporally reduced------------------------
gpm_img, gpm_vis = get_imgcollection(start, end, gee_aoi, 'NASA/GPM_L3/IMERG_MONTHLY_V06', 'precipitation',
                                     function='sum')
mod16_img, mod16_vis = get_imgcollection(start, end, gee_aoi, 'MODIS/006/MOD16A2', 'ET', function='sum')
smap_img, smap_vis = get_imgcollection(start, end, gee_aoi, 'NASA_USDA/HSL/SMAP10KM_soil_moisture',
                                       'ssm')

# download rasters--------------------------
download_img(gpm_img, gee_aoi, 'GPM_P')
download_img(mod16_img, gee_aoi, 'MOD16_ET')
download_img(smap_img, gee_aoi, 'SMAP_SM')

# save tabular data as csv----------------
gpm_df.to_csv('gpm_data.csv')
gpm_agg.to_csv('gpm_water_year.csv')
soil_moisture.to_csv('smap_sm.csv')
mod16_df.to_csv('mod16_et.csv')
#
center_coordinates_of_area = [33.861, -110.712]
Map = geemap.Map(center=center_coordinates_of_area, zoom=7)
Map.addLayer(watershed_gee, {}, 'Watershed')
Map.addLayer(gpm_img, vis_params=gpm_vis, name='GPM Precip')
Map.addLayer(mod16_img, vis_params=mod16_vis, name='MOD16 ET')
Map.addLayer(smap_img, vis_params=smap_vis, name='SMAP soil moisture')
Map.to_html(path_to_save_map)
