CREATE DATABASE weather;

\connect weather;

CREATE TABLE daily(
    date TEXT,
    temp_max FLOAT,
    temp_avg FLOAT,
    temp_min FLOAT,
    dp_max FLOAT,
    dp_avg FLOAT,
    dp_min FLOAT,
    humid_max FLOAT,
    humid_avg FLOAT,
    humid_min FLOAT,
    ws_max FLOAT,
    ws_avg FLOAT,
    ws_min FLOAT,
    press_max FLOAT,
    press_avg FLOAT,
    press_min FLOAT,
    precip FLOAT
);


