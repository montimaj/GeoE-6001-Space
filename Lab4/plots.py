import rasterio
import pandas as pd
import fiona
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import matplotlib.dates as mdates
import datetime
from rasterio.plot import show
from rasterio.mask import mask
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()


def plot_rasters():
    with fiona.open('Watershed/Salt_River.shp', "r") as shapefile:
        geoms = [feature["geometry"] for feature in shapefile]
    gpm_r = rasterio.open('GPM_P.precipitation.tif')
    mod16_r = rasterio.open('MOD16_ET.ET.tif')
    smap_r = rasterio.open('SMAP_SM.ssm.tif')
    gpm_arr, gpm_transform = mask(gpm_r, geoms, crop=True)
    gpm_arr[gpm_arr == 0.] = np.nan
    mod16_arr, mod16_transform = mask(mod16_r, geoms, crop=True)
    mod16_arr[mod16_arr == 0.] = np.nan
    smap_arr, smap_transform = mask(smap_r, geoms, crop=True)
    smap_arr[smap_arr == 0.] = np.nan

    params = {
        'legend.fontsize': 'xx-large',
        'figure.figsize': (15, 6),
        'axes.labelsize': 20,
        'axes.titlesize': 'xx-large',
        'xtick.labelsize': 'xx-large',
        'ytick.labelsize': 'xx-large'
    }
    pylab.rcParams.update(params)

    fig, ax = plt.subplots(1, 1)
    image_hidden = ax.imshow(gpm_arr.squeeze())
    show(gpm_arr, ax=ax, transform=gpm_transform)
    cbar = fig.colorbar(image_hidden, ax=ax)
    cbar.ax.set_ylabel('P (mm/hr)', rotation=-270)
    plt.xlabel('Longitude (Degree)')
    plt.ylabel('Latitude (Degree)')
    plt.show()

    fig, ax = plt.subplots(1, 1)
    image_hidden = ax.imshow(mod16_arr.squeeze())
    show(mod16_arr, ax=ax, transform=mod16_transform)
    cbar = fig.colorbar(image_hidden, ax=ax)
    cbar.ax.set_ylabel('ET (0.1mm/8days)', rotation=-270)
    plt.xlabel('Longitude (Degree)')
    plt.ylabel('Latitude (Degree)')
    plt.show()

    fig, ax = plt.subplots(1, 1)
    image_hidden = ax.imshow(smap_arr.squeeze())
    show(smap_arr, ax=ax, transform=smap_transform)
    cbar = fig.colorbar(image_hidden, ax=ax)
    cbar.ax.set_ylabel('SSM (mm/3 days)', rotation=-270)
    plt.xlabel('Longitude (Degree)')
    plt.ylabel('Latitude (Degree)')
    plt.show()


def plot_csv():
    gpm_monthly_df = pd.read_csv('gpm_data.csv')
    gpm_yearly_df = pd.read_csv('gpm_water_year.csv')
    mod16_df = pd.read_csv('mod16_et.csv')
    smap_df = pd.read_csv('smap_sm.csv')

    # gpm_yearly_df.set_index('water_year').plot(legend=False)
    # plt.xlabel('Year')
    # plt.ylabel('Precipitation (mm)')
    # plt.show()

    gpm_monthly_df.date = pd.to_datetime(gpm_monthly_df.date).dt.strftime('%Y-%m-1 %H:%M:%S')
    mod16_df.date = pd.to_datetime(mod16_df.date)
    smap_df.date = pd.to_datetime(smap_df.date)
    grouper = pd.Grouper(key='date', freq='M')
    mod16_monthly_df = mod16_df.groupby(grouper)['ET'].sum().reset_index()
    mod16_monthly_df.date = pd.to_datetime(mod16_monthly_df.date).dt.strftime('%Y-%m-1 %H:%M:%S')
    smap_monthly_df = smap_df.groupby(grouper)['ssm'].mean().reset_index()
    smap_monthly_df.date = pd.to_datetime(smap_monthly_df.date).dt.strftime('%Y-%m-1 %H:%M:%S')

    gpm_monthly_df.to_csv('Final_GPM_Monthly.csv', index=False)
    mod16_monthly_df.to_csv('Final_MOD16_Monthly.csv', index=False)
    smap_monthly_df.to_csv('Final_SMAP_Monthly.csv', index=False)

    gpm_dt = gpm_monthly_df.date
    mod16_dt = mod16_monthly_df.date
    smap_dt = smap_monthly_df.date
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=12))
    plt.gca().format_xdata = mdates.DateFormatter('%Y-%m')
    plt.plot(gpm_dt, gpm_monthly_df.precipitation)
    plt.plot(mod16_dt, mod16_monthly_df.ET)
    plt.plot(smap_dt, smap_monthly_df.ssm)
    # plt.xlabel('Year')
    plt.legend(['P (mm)', 'ET (mm)', 'SSM (mm)'])
    plt.show()


params = {
        'legend.fontsize': 'xx-large',
        'figure.figsize': (15, 6),
        'axes.labelsize': 20,
        'axes.titlesize': 'xx-large',
        'xtick.labelsize': 'xx-large',
        'ytick.labelsize': 'xx-large',
        'date.epoch': '0000-01-01T00:00:00'
    }
pylab.rcParams.update(params)
plot_csv()
# plot_rasters()



