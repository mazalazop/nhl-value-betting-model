"""Atomic Google Sheets value replacement; the canonical CSV remains authoritative."""
import pandas as pd


def replacement_requests(ws,frame):
    values=[list(frame.columns)]+frame.fillna('').astype(str).values.tolist()
    height=max(ws.row_count,len(values));width=max(ws.col_count,len(frame.columns))
    requests=[]
    if height>ws.row_count or width>ws.col_count:
        requests.append({'updateSheetProperties':{'properties':{'sheetId':ws.id,'gridProperties':{'rowCount':height,'columnCount':width}},'fields':'gridProperties.rowCount,gridProperties.columnCount'}})
    # updateCells with an explicit range clears remaining values in that range in the SAME request.
    requests.append({'updateCells':{'range':{'sheetId':ws.id,'startRowIndex':0,'endRowIndex':height,'startColumnIndex':0,'endColumnIndex':width},
       'rows':[{'values':[{'userEnteredValue':{'stringValue':str(value)}} for value in row]} for row in values],
       'fields':'userEnteredValue'}})
    return requests


def replace_view(ws,frame):
    ws.spreadsheet.batch_update({'requests':replacement_requests(ws,frame)})
