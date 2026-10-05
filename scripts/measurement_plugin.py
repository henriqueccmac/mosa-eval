"""Record collected test cases and outcomes for each full-suite execution."""
import json
import os
from pathlib import Path

def persist():
    Path(os.environ['MEASUREMENT_JSON']).write_text(json.dumps(record, indent=2) + '\n')
record = {'selected': [], 'outcomes': {}, 'collection_errors': []}

def pytest_collection_finish(session):
    record['selected'] = [item.nodeid for item in session.items]
    record['functions'] = len({x.split('[')[0] for x in record['selected']})
    record['cases'] = len(session.items)
    persist()

def pytest_runtest_logreport(report):
    if report.when == 'call' or report.failed or report.skipped:
        outcome = 'xfailed' if report.skipped and hasattr(report, 'wasxfail') else 'xpassed' if report.passed and hasattr(report, 'wasxfail') else report.outcome
        record['outcomes'][report.nodeid] = outcome
        persist()

def pytest_collectreport(report):
    if report.failed:
        record['collection_errors'].append(str(report.longrepr))

def pytest_sessionfinish(session, exitstatus):
    from collections import Counter
    record['exitstatus'] = int(exitstatus)
    record['counts'] = dict(Counter(record['outcomes'].values()))
    Path(os.environ['MEASUREMENT_JSON']).write_text(json.dumps(record, indent=2) + '\n')
