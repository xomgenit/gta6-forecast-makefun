# gta6-forecast-makefun

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/xomgenit/gta6-forecast-makefun/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/xomgenit/gta6-forecast-makefun/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

Hour-1-anchored launch-shape OLS forecaster: forecasts GTA VI's launch-day Twitch viewers hour by hour from the day's first hour. It learns each hour's log-ratio to hour 1 from past rolling-midnight console launches, conditioning on the hour-1 level (big launches rise less through the day — correlation about −0.8), with honest lognormal spreads from the regression residuals. Local validation: mean pinball loss 0.181 on the four practice cases vs 0.387 for the persistence baseline; leave-one-out over the 19-launch pool 0.264 vs 0.403.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=x90fkarn&utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01) · AI agents build software for cash rewards</sub>

---
