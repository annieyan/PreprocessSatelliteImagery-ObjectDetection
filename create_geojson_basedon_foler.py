'''
Create a geojson file which contains labels for bounding boxes  
according to the files in a given folder
'''

import argparse
import os
import geopandas as gpd
import shapely.geometry
import shutil





def get_filenames(abs_dirname):
    
    files = [os.path.join(abs_dirname, f) for f in os.listdir(abs_dirname)]

    i = 0
   # parent_folder = os.path.abspath(abs_dirname + "/../")
   # seperate_subdir = None
    #subdir_name = os.path.join(parent_folder, 'train_small')
    #seperate_subdir = subdir_name
    #os.mkdir(subdir_name)
    name_list = []
    for f in files:
        filename_origin = str(f)
        filename = filename_origin.split('/')[-1]
        #print filename 
        name_list.append(filename)
        # create new subdir if necessary
     #   if i < N:
            #subdir_name = os.path.join(abs_dirname, '{0:03d}'.format(i / N + 1))
           # os.mkdir(subdir_name)
           # seperate_subdir = subdir_name

        # copy file to current dir
            #f_base = os.path.basename(f)
            #shutil.copy(f, os.path.join(subdir_name, f_base))
        i += 1
    return name_list

def geojson_split(geojson_ori, src_dir, savename):
    name_list = set(get_filenames(os.path.abspath(src_dir)))
    gfN = gpd.read_file(geojson_ori) 
    index_list = []
    df_len = len(gfN)
    
    for i in range(0, df_len):
        print('idx', i)
        series_tmp = gfN.loc[i]
        if series_tmp['IMAGE_ID'] in name_list:
            index_list.append(i)
    geometries = [xy for xy in list(gfN.iloc[index_list]['geometry'])]
    crs = {'init': 'epsg:4326'}
    gf = gpd.GeoDataFrame(gfN.iloc[index_list], crs=crs, geometry=geometries)

# geometries = [shapely.geometry.Point(xy) for xy in zip(df.lng, df.lat)]
# gf = gpd.GeoDataFrame(gfN.iloc[0],)
    parent_folder = os.path.abspath(geojson_ori + "/../")
    
    # get folder name 
    f_base = os.path.basename(src_dir)
    save_name = f_base + savename+ '.geojson'
    print('saving file: ', save_name)
    #path = os.path.join(subdir_name, f_base)
    gf.to_file(parent_folder+'/'+ save_name, driver='GeoJSON')
			 
	


def parse_args():
    """Parse command line arguments passed to script invocation."""
    parser = argparse.ArgumentParser(
        description='Split files into multiple subfolders.')

    parser.add_argument('given_dir', help='directory containing 2048 * 2048 image files')
    parser.add_argument('src_geojson', help='source geojson')
  

    return parser.parse_args()


def main():
    """Module's main entry point (zopectl.command)."""
    args = parse_args()
    given_dir= args.given_dir
    geojson_ori = args.src_geojson
    '''
    if not os.path.exists(src_dir):
        raise Exception('Directory does not exist ({0}).'.format(src_dir))
    '''
    # the name the geojson will be 
    savename = '_noblack'
    #get_filenames(os.path.abspath(src_dir))
    geojson_split(os.path.abspath(geojson_ori),os.path.abspath(given_dir), savename)
    #move_files(os.path.abspath(src_dir))
    #seperate_nfiles(os.path.abspath(src_dir))

if __name__ == '__main__':
    main()
