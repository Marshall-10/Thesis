def fix_timestamp_format(ts):
    try:
        date, time = ts.split(' ')
        time_parts = time.split(':')
        if len(time_parts) == 4:  # If there are microseconds
            corrected_time = f"{time_parts[0]}:{time_parts[1]}:{time_parts[2]}.{time_parts[3]}"
        elif len(time_parts) == 3:  # If there are only seconds
            corrected_time = f"{time_parts[0]}:{time_parts[1]}:{time_parts[2]}"
        else:
            raise ValueError("Invalid timestamp format")
        corrected_timestamp = f"{date} {corrected_time}"
        return corrected_timestamp
    except Exception as e:
        print(f"Error processing timestamp: {ts} - {e}")
        return ts
