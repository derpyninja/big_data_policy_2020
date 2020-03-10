import pandas as pd

print("hello")
df = pd.read_excel(io="/Users/felix/ETH/epg-hiwi/MDBs/MDB_Renewables_Finance.xlsx",
                   sheet_name="General", index_col=0, parse_dates=True,
                   convert_float = True)
print(df.head())

for row in df['source'].iteritems():
    try:
        print(row[1].splitlines())
    except AttributeError as e:
        print(e)
        pass