% OBS Data Quality Report - {{ obsName }}
% 
% Prepared on: {{ today }}

---
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

Power spectral density curves are calculated using {{ psdWindowLength }} windows, with {{ psdOverlapPercent }}% overlap.

\newpage{}

# Introduction

{{ introText }}

# General QC

{% if gapList %}
The following gaps/overlaps were observed in the recorded data.

| Channel | Start Time | End Time | Length (s) | Missing Samples |
|:--------|:-----------|:---------|-----------:|----------------:|
{% for gap in gapList %}
| gap.id | gap.start | gap.end | gap.sec | gap.samp |
{% endfor %}

{% else %}
No recording gaps or overlaps were observed in the recorded data.

{% endif %}

\newpage{}

# Seismic Data

{% for ch in seismic_channels %}
## {{ ch.channelName }}
SEED ID: {{ ch.seedID }}

{% if ch.azimuth %}
Orientation: {{ ch.azimuth }} / {{ ch.dip }}
{% endif %}

### Full trace

![Recorded data for channel {{ ch.seedID }}]({{ ch.traceLoc }})

### Spectrogram

![Spectrogram of channel {{ ch.seedID }}]({{ ch.specLoc }})

### Power Spectral Density

![Power spectral density curves for channel {{ ch.seedID }}]({{ ch.psdLoc }})

{% endfor %}

\newpage{}

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
