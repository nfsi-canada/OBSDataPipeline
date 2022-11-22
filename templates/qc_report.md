---
title: OBS Data Quality Report - {{ obsName }}
subtitle: |
          Project: {{ projectName }} \
          Station: {{ stationName }} \
          Location (lat/lon): {{ latString }}, {{ lonString }}
author: 'Prepared on: {{ today }}'
toc-depth: 2
geometry:
- margin=1in
header-includes:
- |
  ```{=latex}
  \usepackage{float}
  \makeatletter
  \def\fps@figure{H}
  \makeatother
  ```
---

\newpage{}

# Summary

OBS name/serial: {{ obsName }} / {{ obsId }}

Project: {{ projectName }}

Station name: {{ stationName }}

Location (lat/lon): {{ latString }}, {{ lonString }}

Water depth (m): {{ waterDepth }}

Deployment date: {{ deployDate }}

Deployment comments: {{ deployComments }}

Recovery date: {{ recoverDate }}

Recovery comments: {{ recoverComments }}

Length of deployment (days): {{ deploymentDays }}

Total clock drift (ms): {{ clockDrift }}

Average power consumption (W): {{ meanPower }}

Remaining battery SOC: {{ batteryLevel }}%

Power spectral density curves are calculated using {{ psdWindowLength }} Hann windows, with {{ psdOverlapPercent }}% overlap.

\newpage{}

# Introduction

{% if intro_pt1 %}
{{ intro_pt1 }}
{% endif %}

{% if introText %}
{{ introText }}
{% endif %}

# General QC

{% if gapList %}
The following gaps/overlaps were observed in the recorded data.

| Channel | Start Time | End Time | Length (s) | Samples |
|:--------|:-----------|:---------|-----------:|--------:|
{% for gap in gapList %}
| {{ gap.id }} | {{ gap.start }} | {{ gap.end }} | {{ gap.sec }} | {{ gap.samp }} |
{% endfor %}

{% else %}
No recording gaps or overlaps were observed in the recorded data.

{% endif %}

{% if centring %}
{{ centring.text }}

![Centring behaviour of OBS {{ obsId }} during deployment]({{ centring.plot }})

{% else %}
Unable to evaluate centring behaviour with available data.

{% endif %}

\newpage{}

# Seismic Data

{% for ch in seismic_channels %}
## {{ ch.channelName }}
SEED ID: {{ ch.seedID }}

{% if ch.azimuth %}
Orientation: {{ ch.azimuth }} / {{ ch.dip }}
{% endif %}

{% if ch.traceLoc %}
### Full trace

![Recorded data for channel {{ ch.seedID }}]({{ ch.traceLoc }})
{% endif %}

{% if ch.specLoc %}
### Spectrogram

{% for spec in ch.specLoc %}
![Spectrogram of channel {{ ch.seedID }} for {{ spec.start }} to {{ spec.end }}]({{ spec.image }})

{% endfor %}
{% endif %}

{% if ch.psdLoc %}
### Power Spectral Density

{% for psd in ch.psdLoc %}
![Power spectral density curves for channel {{ ch.seedID }} for {{ psd.start }} to {{ psd.end }}]({{ psd.image }})

{% endfor %}
{% endif %}

\newpage{}

{% endfor %}

# Oceanographic Data

{% for ch in ocean_channels %}
## {{ ch.channelName }}
SEED ID: {{ ch.seedID }}

![Recorded data for channel {{ ch.seedID }}]({{ ch.traceLoc }})

{% if ch.qcPlotLoc %}
![Range check results for channel {{ ch.seedID }}]({{ ch.qcPlotLoc }})
{% endif %}

{% endfor %}

\newpage{}

# Battery Condition

{% if batteryStats %}
Battery life statistics are calculated from the recorded power consumption and voltage channels. A 3-day rolling window is used, with offset of 1 day between consecutive windows (66% overlap).

This instrument would be expected to enter low-power hibernate mode on or about {{ batteryStats.HibernateEstimate }}.

![Average power consumption, calculated for a 3-day rolling window]({{ batteryStats.meanPowerPlot }})

![Average voltage, calculated for a 3-day rolling window]({{ batteryStats.meanVoltPlot }})

![Voltage gradient, calculated for a 3-day rolling window]({{ batteryStats.gradVoltPlot }})
{% endif %}

{% for ch in power_channels %}
## {{ ch.channelName }}
SEED ID: {{ ch.seedID }}

![Recorded data for channel {{ ch.seedID }}]({{ ch.traceLoc }})

{% if ch.qcPlotLoc %}
![Range check results for channel {{ ch.seedID }}]({{ ch.qcPlotLoc }})
{% endif %}

{% endfor %}

\newpage{}

# Instrument State-of-Health

{% for ch in health_channels %}
## {{ ch.channelName }}
SEED ID: {{ ch.seedID }}

![Recorded data for channel {{ ch.seedID }}]({{ ch.traceLoc }})

{% if ch.qcPlotLoc %}
![Range check results for channel {{ ch.seedID }}]({{ ch.qcPlotLoc }})
{% endif %}

{% endfor %}
