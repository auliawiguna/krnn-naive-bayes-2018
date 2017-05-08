__author__ = 'ahmadauliawiguna'
from texttable import Texttable

table = Texttable()
# table.set_deco(Texttable.HEADER)
table.set_cols_dtype(['t', 't',  't',  't',  't']) # automatic
table.set_cols_align(["l", "r", "r", "r", "l"])
table.add_rows([["Method/Param",    "Data Size", "Data Training Size", "Akurasi", "F-Measure"],
                ["No Samplimg",    "67",    654,   89,    128.001],
                ["Random Oversampling", 67.5434, .654,  89.6,  12800000000000000000000.00023],
                ["kRNN Oversampling",     5e-78,   5e-78, 89.4,  .000000000000128],
                ["SMOTE", .023,    5e+78, 92.,   1280],
                ["krNN Oversampling + SMOTE", .023,    5e+78, 92.,   1280]
                ])
print table.draw()