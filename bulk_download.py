from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from crypto_historical import run

max_pages_per_day = 10

def generate_date_pairs():
    start_date = datetime(2019, 1, 1)
    end_date = datetime.today()
    date_pairs = []

    while start_date < end_date:
        # next_date = start_date + relativedelta(months=1)
        next_date = start_date + relativedelta(days=1)
        if next_date > end_date:
            next_date = end_date
        date_pairs.append((start_date.strftime('%Y-%m-%d'), next_date.strftime('%Y-%m-%d')))
        start_date = next_date

    return date_pairs

# Example usage:
# print(generate_date_pairs())

for start_date, end_date in generate_date_pairs():
    print(f"Start: {start_date}, End: {end_date}")

    # Convert string dates back to datetime objects
    run(start_date=datetime.strptime(start_date, '%Y-%m-%d'), end_date=datetime.strptime(end_date, '%Y-%m-%d'), max_pages=max_pages_per_day)

print("All data downloaded successfully.")