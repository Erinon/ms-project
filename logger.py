from datetime import datetime


FORMAT = '%Y-%m-%d %H:%M:%S.%f'


def debug(msg):
    print(f'{datetime.now().strftime(FORMAT)} - DEBUG - {msg}')

def info(msg):
    print(f'{datetime.now().strftime(FORMAT)} - INFO - {msg}')

def warn(msg):
    print(f'{datetime.now().strftime(FORMAT)} - WARN - {msg}')

def error(msg):
    print(f'{datetime.now().strftime(FORMAT)} - ERROR - {msg}')
