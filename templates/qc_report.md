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

Water depth (m): {{ depthString }}

Deployment date: {{ deployDate }}

Deployment comments: {{ deployComments }}

Recovery date: {{ recoverDate }}

Recovery comments: {{ recoverComments }}

Length of deployment (days): {{ deploymentDays }} | At seafloor: {{ seafloorDays }}

Total clock drift (ms): {{ clockDrift }} ({{ clockDriftPerDay }} ms/day)

Average power consumption (W): {{ meanPower }}

Battery SOC: At deployment: {{ batteryLevel.start }}% | Remaining: {{ batteryLevel.end }}%

{% if tiltAtRecovery %}
Tilt from vertical at recovery: {{ tiltAtRecovery }} degrees

{% endif %}
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

{% if channelList %}
Recorded data channels:

| Channel | Start Time | End Time | Sampling Rate (Hz) |
|:--|:---|:---|-:|
{% for ch in channelList %}
| {{ ch.id }} | {{ ch.start }} | {{ ch.end }} | {{ ch.sampling }} |
{% endfor %}

{% endif %}

{% if gapList %}
The following gaps/overlaps were observed in the recorded data.

| Channel | Start Time | End Time | Length (s) | Samples |
|:--|:---|:---|-:|-:|
{% for gap in gapList %}
| {{ gap.id }} | {{ gap.start }} | {{ gap.end }} | {{ gap.sec }} | {{ gap.samp }} |
{% endfor %}

{% else %}
No recording gaps or overlaps were observed in the recorded data.

{% endif %}

{% if humid %}
Abnormal change(s) in humidity were observed during this deployment.

![Above normal changes in humidity data {{ humid.ch }} during deployment]({{ humid.plot }})

| Start Time | End Time | Length (s) | Deviation (%Rh) |
|:-|:-|-:|-:|
{% for h in humid.triggers %}
| {{ h.start }} | {{ h.end }} | {{ h.sec }} | {{ h.dev }} |
{% endfor %}

{% endif %}

{% if centring %}
{{ centring.text }}

![Centring behaviour of OBS {{ obsId }} during deployment]({{ centring.plot }})

{% endif %}

\newpage{}

# Seismic Data

{% for ch in seismic_channels %}
## {{ ch.channelName }}
SEED ID: {{ ch.seedID }}

{% if ch.traceLoc %}
### Full trace

![Recorded data for channel {{ ch.seedID }}]({{ ch.traceLoc }})
{% endif %}

{% if ch.specLoc %}
### Spectrogram

{% for spec in ch.specLoc %}
{% if ch.hydrophone %}
![PSD spectrogram of channel {{ ch.seedID }} for {{ spec.start }} to {{ spec.end }}]({{ spec.image }})
{% else %}
![Acceleration PSD spectrogram of channel {{ ch.seedID }} for {{ spec.start }} to {{ spec.end }}]({{ spec.image }})
{% endif %}

{% endfor %}
{% endif %}

{% if ch.psdLoc %}
### Power Spectral Density

{% for psd in ch.psdLoc %}
{% if ch.hydrophone %}
![Power spectral density curves for channel {{ ch.seedID }} for {{ psd.start }} to {{ psd.end }}]({{ psd.image }})
{% else %}
![Acceleration power spectral density curves for channel {{ ch.seedID }} for {{ psd.start }} to {{ psd.end }}]({{ psd.image }})
{% endif %}

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
Battery life statistics are calculated from the recorded power consumption and voltage channels. A {{ batteryStats.window_str }} rolling window is used, with 66% overlap between consecutive windows.

This instrument would be expected to enter low-power hibernate mode on or about {{ batteryStats.HibernateEstimate }}.

![Average power consumption, calculated for a {{ batteryStats.window_str }} rolling window]({{ batteryStats.meanPowerPlot }})

![Average voltage, calculated for a {{ batteryStats.window_str }} rolling window]({{ batteryStats.meanVoltPlot }})

![Voltage gradient, calculated for a {{ batteryStats.window_str }} rolling window]({{ batteryStats.gradVoltPlot }})

{% if batteryStats.currentPlot %}
![Average current draw, calculated from recorded voltage and power consumption, for a {{ batteryStats.window_str }} rolling window]({{ batteryStats.currentPlot }})

{% endif %}
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
