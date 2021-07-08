# Santization & Manual Pre-Processing 
```from raw data to user_by_month & all_by_month``` \
Modified: March 31st, 2021 \
-- jeffmur

## Change Log
---
| Date | Note |
| ---  | ---|
| 3/9  | Added .yaml for anaconda environment + preprocess.py |
| 3/23 | Added Pre-Processing Lib for MDC Sanitization

For city limits see **FetchGeoLocation( *"city, country"* )** except geoLife

## Privamov
---
``` Lyon, France ``` \
City Limits = ['45.7073666', '45.8082628', '4.7718134', '4.8983774'] \ 
Number of Users: 116

**Note**: raw-* files must be formatted to correct delimeters, currently: indented, TODO: "," 

## GeoLife 
---
``` Beijing, China ``` \
City Limits = ['39.7350200', '40.1073889', '116.1644800', '116.6597843']
Number of Users: 182

**Note**: Does NOT work with fetchGeoLocation(). Must be manually set (due to change 22 days ago)

## MDC
---
```Lausanne, District de Lausanne, Vaud, Switzerland``` \
City Limits = ['46.5043006', '46.6025773', '6.5838681', '6.7208137'] \
Number of Users: 185

**Note**: raw files must be formatted to correct delimeters and sanitized. 

Used Datasets: records.csv (9.5 GB), gps.csv (1.3 GB). 

Please read through sanitize.py before executing. Usage of bash and python to setup gps-sanitized.csv & gpsRecords.csv

**Order of Operation for MDC**
1. sanitize.py 
    - Input  : Raw Datasets
    - Output : Clean, formatted csv for Pandas Dataframes
    - Note   : Use of bash + python to specify GPS records only
2. split_by_month.py 
    - Input  : Pandas Dataframe (sanitize output)
    - Output : Directory of all months csvs
3. split_user_months.py
    - Input : All month csvs in directory
    - Output: Split months by user id
