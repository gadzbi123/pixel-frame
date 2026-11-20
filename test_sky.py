import datetime as dt
import unittest

import sky

GREENWICH = (51.48, 0.0)


def utc(text):
    return dt.datetime.fromisoformat(text).replace(tzinfo=dt.timezone.utc).timestamp()


class FakeFeed:
    current, fetched = sky.Weather(), 0.

    def weather(self):
        return self.current


class MoonTests(unittest.TestCase):
    def test_full_at_the_lunar_eclipse_of_march_2026(self):
        self.assertGreater(sky.moon_at(utc("2026-03-03T11:33"), GREENWICH).illumination, .999)

    def test_new_at_the_solar_eclipse_of_august_2026(self):
        moon = sky.moon_at(utc("2026-08-12T17:46"), GREENWICH)
        self.assertLess(moon.illumination, .001)
        # The eclipse is total only far north (gamma 0.9), so the centres are 0.9 degrees apart.
        self.assertAlmostEqual(180 - moon.phase, .9, delta=.15)

    def test_waxing_before_full_and_waning_after(self):
        self.assertTrue(sky.moon_at(utc("2026-02-24T12:00"), GREENWICH).waxing)
        self.assertFalse(sky.moon_at(utc("2026-03-10T12:00"), GREENWICH).waxing)
        quarter = sky.moon_at(utc("2026-02-24T12:00"), GREENWICH).illumination
        self.assertTrue(.3 < quarter < .7)

    def test_an_evening_crescent_is_lit_on_the_side_of_the_set_sun(self):
        moon = sky.moon_at(utc("2026-02-20T17:45"), GREENWICH)
        self.assertGreater(moon.altitude, 10)
        self.assertTrue(200 < moon.azimuth < 270)    # south-west
        self.assertTrue(100 < moon.bright_side < 180)  # lit side down and to the right

    def test_the_sun_at_noon_on_the_solstice_and_the_equinox(self):
        altitude, azimuth = sky.sun_position(utc("2026-06-21T12:02"), GREENWICH)
        self.assertAlmostEqual(altitude, 90 - 51.48 + 23.44, delta=.3)
        self.assertAlmostEqual(azimuth, 180, delta=1)
        self.assertAlmostEqual(sky.sun_position(utc("2026-03-20T12:07"), GREENWICH)[0], 90 - 51.48, delta=.5)


class WeatherTests(unittest.TestCase):
    def test_open_meteo_reports(self):
        rain = sky.weather_from({"weather_code": 63, "cloud_cover": 100, "wind_speed_10m": 0, "wind_direction_10m": 0})
        self.assertEqual((rain.rain, rain.snow, rain.cloud, rain.storm), (.65, 0, 1, False))
        self.assertEqual(sky.weather_from({"weather_code": 1, "cloud_cover": 20}), sky.Weather())
        self.assertEqual(sky.weather_from({"weather_code": 45, "cloud_cover": 100}).fog, .7)
        self.assertTrue(sky.weather_from({"weather_code": 95, "cloud_cover": 100}).storm)
        snow = sky.weather_from({"weather_code": 71, "cloud_cover": 50})
        self.assertEqual((snow.snow, snow.cloud), (.35, .85))  # falling snow means a grey sky

    def test_wind_carries_rain_away_from_where_it_blows_from(self):
        # The frame looks south: a west wind blows towards the east, on the left.
        west = sky.weather_from({"weather_code": 61, "wind_speed_10m": 50, "wind_direction_10m": 270})
        east = sky.weather_from({"weather_code": 61, "wind_speed_10m": 25, "wind_direction_10m": 90})
        self.assertEqual((west.wind, east.wind), (-1, .5))

    def test_weather_comes_on_gradually(self):
        sky_ = sky.Sky(weather=sky.Weather())
        self.assertEqual(sky_.weather(0), sky.Weather())
        sky_.target = sky.PRESETS["rain"]
        amounts = [sky_.weather(second).rain for second in range(0, 600, 30)]
        self.assertEqual(amounts, sorted(amounts))
        self.assertLess(amounts[1], .3)
        self.assertEqual(amounts[-1], sky.PRESETS["rain"].rain)

    def test_weather_arrives_in_full_at_the_frame_rate(self):
        sky_ = sky.Sky(weather=sky.Weather())
        sky_.weather(0)
        sky_.target = sky.Weather(cloud=.9)
        for tick in range(1, 8*600):  # ten minutes at 8 frames a second
            shown = sky_.weather(tick/8)
        self.assertEqual(shown.cloud, .9)

    def test_the_first_report_shows_at_once(self):
        # The frame starts before the report is in: rain must not fade in from a clear sky after every restart.
        sky_ = sky.Sky()
        sky_.feed = feed = FakeFeed()
        self.assertEqual(sky_.weather(0), sky.Weather())
        feed.current, feed.fetched = sky.PRESETS["rain"], 1.
        self.assertEqual(sky_.weather(1), sky.PRESETS["rain"])
        feed.current = sky.Weather()
        self.assertGreater(sky_.weather(2).rain, 0)  # later reports still come on gradually

    def test_without_a_place_there_is_no_moon_and_no_feed(self):
        sky_ = sky.Sky()
        self.assertIsNone(sky_.moon(0))
        self.assertIsNone(sky_.feed)
        self.assertEqual(sky_.weather(0), sky.Weather())


if __name__ == "__main__":
    unittest.main()
