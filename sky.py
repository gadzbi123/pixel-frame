"""The moon and the weather over the frame's location.

The moon follows Paul Schlyter's low-precision method ("How to compute
planetary positions"), good to a few arc minutes, which is plenty for a moon
on a painting. The weather comes from Open-Meteo, which is free and needs no
account. Nothing here draws; pixel_frame.py does.
"""

from __future__ import annotations

import json
import math
import sys
import threading
import time
import urllib.request
from dataclasses import dataclass, replace

# Astronomy ---------------------------------------------------------------


def _sin(degrees: float) -> float:
    return math.sin(math.radians(degrees))


def _cos(degrees: float) -> float:
    return math.cos(math.radians(degrees))


def _atan2(y: float, x: float) -> float:
    return math.degrees(math.atan2(y, x))


def _days(when: float) -> float:
    """Days since 2000 January 0.0 UT, the epoch of Schlyter's elements."""
    return when/86400 - 10956


def _sun(d: float) -> tuple[float, float, float]:
    """The sun's ecliptic longitude, its mean anomaly and its mean longitude, in degrees."""
    w = 282.9404 + 4.70935e-5*d
    e = .016709 - 1.151e-9*d
    anomaly = (356.0470 + .9856002585*d) % 360
    eccentric = anomaly + math.degrees(e)*_sin(anomaly)*(1 + e*_cos(anomaly))
    v = _atan2(math.sqrt(1 - e*e)*_sin(eccentric), _cos(eccentric) - e)
    return (v + w) % 360, anomaly, (anomaly + w) % 360


def _moon(d: float) -> tuple[float, float, float]:
    """The moon's geocentric ecliptic longitude and latitude in degrees, and its distance in Earth radii."""
    node = 125.1228 - .0529538083*d
    tilt, w, a, e = 5.1454, 318.0634 + .1643573223*d, 60.2666, .054900
    anomaly = (115.3654 + 13.0649929509*d) % 360
    eccentric = anomaly + math.degrees(e)*_sin(anomaly)*(1 + e*_cos(anomaly))
    for _ in range(3):
        eccentric -= (eccentric - math.degrees(e)*_sin(eccentric) - anomaly)/(1 - e*_cos(eccentric))
    x, y = a*(_cos(eccentric) - e), a*math.sqrt(1 - e*e)*_sin(eccentric)
    v, r = _atan2(y, x), math.hypot(x, y)
    xh = r*(_cos(node)*_cos(v + w) - _sin(node)*_sin(v + w)*_cos(tilt))
    yh = r*(_sin(node)*_cos(v + w) + _cos(node)*_sin(v + w)*_cos(tilt))
    zh = r*_sin(v + w)*_sin(tilt)
    longitude, latitude = _atan2(yh, xh), _atan2(zh, math.hypot(xh, yh))
    # The largest perturbations, mostly from the sun.
    _, sun_anomaly, sun_longitude = _sun(d)
    mean = node + w + anomaly
    elongation, argument = mean - sun_longitude, mean - node
    m, ms, dd, f = anomaly, sun_anomaly, elongation, argument
    longitude += (-1.274*_sin(m - 2*dd) + .658*_sin(2*dd) - .186*_sin(ms) - .059*_sin(2*m - 2*dd)
                  - .057*_sin(m - 2*dd + ms) + .053*_sin(m + 2*dd) + .046*_sin(2*dd - ms)
                  + .041*_sin(m - ms) - .035*_sin(dd) - .031*_sin(m + ms) - .015*_sin(2*f - 2*dd)
                  + .011*_sin(m - 4*dd))
    latitude += (-.173*_sin(f - 2*dd) - .055*_sin(m - f - 2*dd) - .046*_sin(m + f - 2*dd)
                 + .033*_sin(f + 2*dd) + .017*_sin(2*m + f))
    r += -.58*_cos(m - 2*dd) - .46*_cos(2*dd)
    return longitude % 360, latitude, r


def _horizontal(d: float, longitude: float, latitude: float, place: tuple[float, float]) -> tuple[float, float]:
    """Altitude and azimuth (from north through east) of an ecliptic position, seen from a place."""
    tilt = 23.4393 - 3.563e-7*d
    x, y, z = _cos(longitude)*_cos(latitude), _sin(longitude)*_cos(latitude), _sin(latitude)
    y, z = y*_cos(tilt) - z*_sin(tilt), y*_sin(tilt) + z*_cos(tilt)
    ascension, declination = _atan2(y, x), _atan2(z, math.hypot(x, y))
    _, _, sun_longitude = _sun(d)
    hours = (d % 1)*24
    sidereal = sun_longitude + 180 + hours*15 + place[1]
    angle = sidereal - ascension
    x, y, z = _cos(angle)*_cos(declination), _sin(angle)*_cos(declination), _sin(declination)
    lat = place[0]
    xh, zh = x*_sin(lat) - z*_cos(lat), x*_cos(lat) + z*_sin(lat)
    return math.degrees(math.asin(max(-1, min(1, zh)))), (_atan2(y, xh) + 180) % 360


@dataclass(frozen=True)
class Moon:
    """The moon as a place sees it."""
    altitude: float      # degrees above the horizon, parallax included
    azimuth: float       # degrees from north through east
    phase: float         # 0 full, 180 new: the angle sun-moon-earth
    bright_side: float   # which way its lit side faces, degrees clockwise from straight up
    waxing: bool

    @property
    def illumination(self) -> float:
        """The lit share of the disc, 0 to 1."""
        return (1 + _cos(self.phase))/2


def sun_position(when: float, place: tuple[float, float]) -> tuple[float, float]:
    """The sun's altitude and azimuth in degrees, from a Unix time and (latitude, longitude)."""
    d = _days(when)
    return _horizontal(d, _sun(d)[0], 0, place)


def moon_at(when: float, place: tuple[float, float]) -> Moon:
    d = _days(when)
    sun_longitude = _sun(d)[0]
    longitude, latitude, distance = _moon(d)
    altitude, azimuth = _horizontal(d, longitude, latitude, place)
    altitude -= math.degrees(math.asin(1/distance))*_cos(altitude)  # we look from the surface, not the centre
    elongation = math.degrees(math.acos(_cos(longitude - sun_longitude)*_cos(latitude)))
    sun_altitude, sun_azimuth = _horizontal(d, sun_longitude, 0, place)
    # Direction from the moon to the sun along the sky, as a compass bearing with "up" for north.
    turn = sun_azimuth - azimuth
    bright_side = _atan2(_sin(turn)*_cos(sun_altitude),
                         _cos(altitude)*_sin(sun_altitude) - _sin(altitude)*_cos(sun_altitude)*_cos(turn))
    return Moon(altitude, azimuth, 180 - elongation, bright_side % 360,
                (longitude - sun_longitude) % 360 < 180)


# Weather -----------------------------------------------------------------

@dataclass(frozen=True)
class Weather:
    """What to show, each 0 to 1."""
    rain: float = 0
    snow: float = 0
    fog: float = 0
    cloud: float = 0      # how grey the sky is
    storm: bool = False   # lightning
    wind: float = 0       # sideways drift of rain and snow on screen, -1 (left) to 1 (right)

    def describe(self) -> str:
        parts = [f"{name} {amount:.0%}" for name, amount in
                 (("rain", self.rain), ("snow", self.snow), ("fog", self.fog), ("cloud", self.cloud)) if amount]
        return ", ".join(parts + ["lightning"]*self.storm) or "clear"


# WMO weather codes as Open-Meteo reports them: (rain, snow, fog, storm).
_CODES = {45: (0, 0, .7, False), 48: (0, 0, .85, False),
          51: (.2, 0, 0, False), 53: (.3, 0, 0, False), 55: (.4, 0, 0, False),
          56: (.25, 0, 0, False), 57: (.4, 0, 0, False),
          61: (.4, 0, 0, False), 63: (.65, 0, 0, False), 65: (1, 0, 0, False),
          66: (.4, 0, 0, False), 67: (.8, 0, 0, False),
          71: (0, .35, 0, False), 73: (0, .65, 0, False), 75: (0, 1, 0, False), 77: (0, .25, 0, False),
          80: (.45, 0, 0, False), 81: (.7, 0, 0, False), 82: (1, 0, 0, False),
          85: (0, .5, 0, False), 86: (0, .9, 0, False),
          95: (.8, 0, 0, True), 96: (.9, 0, 0, True), 99: (1, 0, 0, True)}

PRESETS = {"clear": Weather(),
           "cloudy": Weather(cloud=.9),
           "fog": Weather(fog=.8, cloud=.6),
           "drizzle": Weather(rain=.3, cloud=.85),
           "rain": Weather(rain=.7, cloud=.9, wind=.2),
           "snow": Weather(snow=.7, cloud=.85, wind=.1),
           "storm": Weather(rain=1, cloud=1, storm=True, wind=.4)}


def weather_from(current: dict) -> Weather:
    """Read Open-Meteo's "current" block."""
    rain, snow, fog, storm = _CODES.get(int(current.get("weather_code", 0)), (0, 0, 0, False))
    cloud = min(1, max(0, (float(current.get("cloud_cover", 0)) - 30)/70))
    if rain or snow:
        cloud = max(cloud, .85)
    # Wind blows away from where it comes from. The frame looks south, so east is on the left.
    speed, heading = float(current.get("wind_speed_10m", 0)), float(current.get("wind_direction_10m", 0))
    wind = min(1, speed/50)*_sin(heading)
    return Weather(rain, snow, fog, cloud, storm, round(wind, 2))


class WeatherFeed:
    """The weather at a place, fetched in the background every quarter of an hour."""

    URL = ("https://api.open-meteo.com/v1/forecast?latitude={:.2f}&longitude={:.2f}"
           "&current=weather_code,cloud_cover,wind_speed_10m,wind_direction_10m")
    EVERY, RETRY, STALE = 900, 60, 3*3600

    def __init__(self, place: tuple[float, float]):
        self.place = place
        self.current, self.fetched = Weather(), 0.0
        threading.Thread(target=self._run, daemon=True).start()

    def weather(self) -> Weather:
        """The latest report, or clear sky once it is hours old."""
        return self.current if time.time() - self.fetched < self.STALE else Weather()

    def _run(self) -> None:
        while True:
            try:
                request = urllib.request.Request(self.URL.format(*self.place), headers={"User-Agent": "pixel-frame"})
                with urllib.request.urlopen(request, timeout=20) as response:
                    weather = weather_from(json.load(response)["current"])
            except Exception as error:  # no network yet, or the service is down: keep the last report
                print(f"Weather unavailable: {error}", file=sys.stderr, flush=True)
                time.sleep(self.RETRY if not self.fetched else self.EVERY)
                continue
            if weather != self.current or not self.fetched:
                print(f"{time.strftime('%H:%M')} weather: {weather.describe()}", flush=True)
            self.current, self.fetched = weather, time.time()
            time.sleep(self.EVERY)


class Sky:
    """The moon and the weather for the display, eased so that changes come on gradually."""

    EASE = 90  # seconds for weather to come on or clear

    def __init__(self, place: tuple[float, float] | None = None, weather: Weather | None = None):
        self.place = place
        self.feed = WeatherFeed(place) if place and weather is None else None
        self.target = weather or Weather()
        self.shown, self.updated = self.target, None

    def moon(self, when: float) -> Moon | None:
        return moon_at(when, self.place) if self.place else None

    def weather(self, now: float) -> Weather:
        """The weather on screen, a step closer to the latest report on each call."""
        target = self.feed.weather() if self.feed else self.target
        step = 1 if self.updated is None else min(1, max(0, now - self.updated)/self.EASE)
        # Until the first report is in, nothing eases: it shows at once when it comes.
        self.updated = now if self.feed is None or self.feed.fetched else None
        shown = self.shown

        def ease(name: str) -> float:
            goal = getattr(target, name)
            value = getattr(shown, name) + (goal - getattr(shown, name))*step
            return value if abs(value - goal) > .005 else goal
        self.shown = replace(target, **{name: ease(name) for name in ("rain", "snow", "fog", "cloud", "wind")})
        return self.shown
