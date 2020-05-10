from __future__ import print_function
import sys
import pprint
import numpy as np
import savReaderWriter as sav

spss_file = "./test_data/gerhard.sav"
spss_file_out = "./test_data/gerhard_out.sav"

ioLocale = "german" if sys.platform.startswith("win") else "de_DE.cp1252"

data = sav.SavReader(spss_file, returnHeader=True, ioUtf8=True, ioLocale=ioLocale, rawMode=True)
with data:
    allData = data.all()
    variables = allData[0]
    records = allData[1:]
    print(data.varTypes["name"] == len(records[0][1]))   # 24 while it should be 20! -->> rawMode: strings are ceiled multiples of 8

    formats = ["S%d" % data.varTypes[v] if data.varTypes[v] else np.float64 for v in data.varNames]
    dtype = np.dtype({'names': data.varNames, 'formats': formats})
    structured_array = np.array([tuple(record) for record in records], dtype=dtype)


allDataArray = np.array(records)  # in the most recent version one can directly read to numpy arrays
print(records)

# reading metadata from SPSS file
with sav.SavHeaderReader(spss_file, ioUtf8=True, ioLocale=ioLocale) as header:
    metadata = header.dataDictionary(asNamedtuple=False)  # Why does this take so long?

pprint.pprint(metadata)

# writing unmodified data
with sav.SavWriter(spss_file_out, overwrite=True, ioUtf8=True, ioLocale=ioLocale,
                   mode=b'wb', refSavFileName=None, **metadata) as writer:
    for i, record in enumerate(structured_array):
        writer.writerow(record)

