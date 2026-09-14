"""
Author: Marco Schmid
Affiliation: Institute for Industrial Ecology - Pforzheim University
File: merit-order_auto-fill.py
Description: Module to fill LAEND/oemof scenario excel with computed merit order.
"""


from pathlib import Path

import openpyxl

# --- File ---
TARGET_FILE = "validation_2024_016_lignite2.xlsx"   # file to be edited
SHEET = "commodity_sources"

# Columns (1-based): A=1 label, G=7 variable_costs, K=11 max,
# L=12 min, M=13 startup_costs, N=14 shutdown_costs,
# O=15 minimum_uptime, P=16 minimum_downtime, Q=17 initial_status, R=18 comment
FIRST_ROW = 19          # first SEE row

# German comment values -> English fuel token
FUEL_DE_EN = {
    "Braunkohle":         "lignite",
    "Steinkohle":         "hardcoal",
    "Erdgas":             "naturalgas",
    "Mineralölprodukte":  "oil",
}

# Unit commitment parameters per fuel -> (min, startup_factor, shutdown, uptime, downtime, init)
# startup_costs = max * startup_factor; naturalgas/oil remain empty
UC_PARAMS = {
    "lignite":  dict(min=0.5, startup_factor=35*1.32, shutdown=0, uptime=6, downtime=6, init=0),
    "hardcoal": dict(min=0.3, startup_factor=45*1.32, shutdown=0, uptime=5, downtime=4, init=0),
}

# --- Edit target file ---
wb = openpyxl.load_workbook(
    Path(__file__).resolve().parent.parent / TARGET_FILE
    )
ws = wb[SHEET]

relabeled = 0
for row in range(FIRST_ROW, ws.max_row + 1):
    label = ws.cell(row=row, column=1).value
    if not label or not str(label).startswith("SEE"):
        continue

    see_id     = str(label)
    price      = ws.cell(row=row, column=7).value            # G variable_costs
    max_mw     = ws.cell(row=row, column=11).value           # K max (already populated)
    comment_de = str(ws.cell(row=row, column=18).value or "")  # R comment

    fuel = FUEL_DE_EN.get(comment_de, "unknown")
    max_str = "" if max_mw is None else str(max_mw)

    # Label: resource_<SEE-ID>_<max><fuel>_<price>EUR
    ws.cell(row=row, column=1).value = f"resource_{see_id}_{max_str}{fuel}_{price}EUR"

    # Columns L-Q
    p = UC_PARAMS.get(fuel)
    if p:
        ws.cell(row=row, column=12).value = p["min"]                          # L min
        ws.cell(row=row, column=13).value = f"=K{row}*{p['startup_factor']}"  # M startup_costs (formula)
        ws.cell(row=row, column=14).value = p["shutdown"]                     # N shutdown_costs
        ws.cell(row=row, column=15).value = p["uptime"]                       # O minimum_uptime
        ws.cell(row=row, column=16).value = p["downtime"]                     # P minimum_downtime
        ws.cell(row=row, column=17).value = p["init"]                         # Q initial_status
    else:
        for col in range(12, 18):   # naturalgas / oil: L-Q empty
            ws.cell(row=row, column=col).value = None
    ws.cell(row=row, column=18).value = ""  # R comment deletion

    relabeled += 1

wb.save(
    Path(__file__).resolve().parent.parent / TARGET_FILE
    )
print(f"{relabeled} SEE rows renamed and filled.")