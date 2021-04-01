# Joshua Sterner
# Modifed: Jeffrey Murray 
from pathlib import Path

def split_trajectory(trajectory):
    rows = []
    month = None
    with open(trajectory, "r") as t:
        for line in t:
            if(line.startswith(",")): # Skip Header
                continue
            # year, month, day
            date = line.split(',')[2].split('-')
            next_month = (int(date[0]), int(date[1]))
            if month == None:
                month = next_month
            if month != next_month:
                yield month, rows
                month = next_month
                rows = []
            rows.append(line.strip())
        yield month, rows

def split_by_month(trajectory):
    data = {}
    for month, rows in split_trajectory(trajectory):
        if month not in data:
            data[month] = rows
        else:
            data[month] = data[month] + rows
    return data

# TODO: PATH @ TO CHANGE OF INPUT FILE & OPTIONAL output_directory

def main():
    output_dir = Path('mdcd_by_month')
    output_dir.mkdir()

    in_file = Path('/home/jeffmur/dev/mdcd/gpsData/gps-sanitized.csv')

    data = split_by_month(in_file)
    for year, month in data.keys():
        rows = data[year, month]
        output_file = output_dir / Path(f'{year}_{month:02}.csv')
        with output_file.open(mode='w') as f:
            f.write('\n'.join(rows)+'\n')
            
if __name__ == '__main__':
    main()
