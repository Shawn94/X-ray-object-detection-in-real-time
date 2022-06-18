import glob
import os


while(True):
    list_of_png = glob.glob('converted/*.png') # * means all if need specific format then *.csv
    last_added_file = max(list_of_png, key=os.path.getctime)
    print(list_of_png)

