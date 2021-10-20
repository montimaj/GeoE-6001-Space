import ee
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import rasterio
from rasterio.plot import show
import geopandas as gpd
import requests
from tqdm import tqdm
import zipfile
import os
import pandas as pd
from glob import glob

# initialize google earth engine if it hasn't been initialized yet
if not ee.data._credentials:
    ee.Authenticate()
    ee.Initialize()

# functions for use in the lab

def ee_imgcoll_to_df(imagecollection, lat,lon):
    """Transforms client-side ee.Image.getRegion array to pandas.DataFrame."""
    poi = ee.Geometry.Point(lon, lat)
    arr = imagecollection.getRegion(poi,500).getInfo()

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

# load prism data
def get_prism_image(date1,date2,geometry):
  prism = ee.ImageCollection('OREGONSTATE/PRISM/AN81m')
  prism_img = prism.filterDate(date1,date2).select('ppt').sum().clip(geometry)
  return(prism_img) # returns prism total precipitation, in mm

def get_elev(geometry):
  elev = ee.Image('USGS/NED').clip(geometry)
  return(elev)

def get_srtm(geometry):
  elev = ee.Image('USGS/SRTMGL1_003').clip(geometry)
  return(elev)

def get_gpm_image(date1,date2,geometry):
  gpm = ee.ImageCollection('NASA/GPM_L3/IMERG_MONTHLY_V06')
  gpm_img = gpm.filterDate(date1,date2).select('precipitation').sum().multiply(24*365/12).clip(geometry)
  return(gpm_img) # returns total gpm precipitation in mm

def get_mod16ET(date1,date2,geometry):
  mod16 = ee.ImageCollection('MODIS/006/MOD16A2')
  mod16_img = mod16.filterDate(date1,date2).select('ET').sum().divide(10).clip(geometry)
  return(mod16_img)

def get_mod16PET(date1,date2,geometry):
  mod16 = ee.ImageCollection('MODIS/006/MOD16A2')
  mod16_img = mod16.filterDate(date1,date2).select('PET').sum().divide(10).clip(geometry)
  return(mod16_img)

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
    
def download_ssebop_data(start, end, outdir='SSEBop'):
    """
    Download SSEBop Data
    :param sse_link: Main SSEBop link without file name
    :param year_list: List of years
    :param start_month: Start month in %m format
    :param end_month: End month in %m format
    :param outdir: Download directory
    :return: None
    """
    
    if os.path.isdir(outdir)==False:
        os.mkdir(outdir)
    
    sse_link = 'https://edcintl.cr.usgs.gov/downloads/sciweb1/shared/uswem/web/conus/eta/modis_eta/monthly/downloads/'
    date_list = pd.date_range(start,end,freq='MS')
    
    for kk in range(len(date_list)):
        date = str(date_list[kk])
        yr = date[0:4]
        mth = date[5:7]
        url = sse_link + 'm' + yr + mth + '.zip'
        local_file_name = outdir + '/SSEBop_' + yr + mth + '.zip'
        fname = outdir+'/m'+yr+mth+'.modisSSEBopETactual.tif'
        if os.path.isfile(fname)==False:
            print('downloading SSEBop scene: '+ fname)
            r = requests.get(url, allow_redirects=True)
            open(local_file_name, 'wb').write(r.content)    
            with zipfile.ZipFile(local_file_name, 'r') as zip_ref:
                zip_ref.extractall(outdir)
            os.remove(local_file_name)
        
def get_ssebop_time_series_point(lat,lon,ssebop_dir='SSEBop'):
    files = glob(ssebop_dir+'/*SSEBop*.tif')
    dates = files.copy()
    vals = np.zeros(len(dates))
    for kk in range(len(dates)):
        dates[kk] = files[kk].split('\\')[-1][1:5]+'-'+files[kk].split('\\')[-1][5:7]+'-15'
        r = rasterio.open(files[kk])
        for val in r.sample([(lon,lat)]):
            vals[kk] = val
    df = pd.DataFrame({'Date':dates,'ET':vals})
    df['Date'] = pd.to_datetime(df['Date'])
    return(df)
    
plt.close('all')

# read list of sites
sites = pd.read_csv('eddy_covariance_sites.txt')
files = glob('*AMF*.csv')

download_ssebop_data('2015-01-01','2021-07-01')

for kk in range(len(files)):
    file= files[kk]
    sitename = files[kk].split('_')[1]
    filt = sites.site_name.str.contains(sitename)
    lat = np.array(sites.lat[filt])[0]
    lon = np.array(sites.lon[filt])[0]
    eddy_data = pd.read_csv(file,skiprows=2,
                                  parse_dates=['TIMESTAMP_START'],na_values=-9999)    
    # the column 'LE' has latent heat flux data. How can you convert this to daily ET in mm?
    conversion_factor=3600*24/2260/1000 # change this to some value that makes sense!
    eddy_data['ET'] = eddy_data.LE*conversion_factor
    eddy_ET = eddy_data[['TIMESTAMP_START','ET']]
    # rice_eddy_ET = rice_eddy_ET.dropna()
    
    eddy_ET = eddy_ET.set_index('TIMESTAMP_START')
    eddy_daily = eddy_ET.resample('W').mean()
    
    ssebop_df = get_ssebop_time_series_point(lat, lon)
    
    mod16_image_collection =  ee.ImageCollection('MODIS/006/MOD16A2').filterDate(np.min(eddy_data['TIMESTAMP_START']),np.max(eddy_data['TIMESTAMP_START']))
    mod16_df = ee_imgcoll_to_df(mod16_image_collection,lat, lon) # image collection, lat, long

    site_description = np.array(sites.site_name[filt])[0]+', '+np.array(sites['crop type'][filt])[0]
    
    plt.figure();plt.plot(eddy_daily);plt.title(site_description)
    plt.plot(ssebop_df['Date'],ssebop_df['ET']/(365/12)) # SSEBop daily ET, mm
    plt.plot(mod16_df['datetime'],mod16_df['ET']/10/8) # MOD16 daily ET, mm
    plt.xlim([np.min(eddy_data['TIMESTAMP_START']),np.max(eddy_data['TIMESTAMP_START'])])
    plt.legend(['Eddy Covariance ET','SSEBop ET','MOD16 ET'])

