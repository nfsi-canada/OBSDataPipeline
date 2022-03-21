# OBS Data Quality Report - {{ stationName }}

## Summary

Project: {{ projectName }}

Station name: {{ stationName }}

Location (lat/lon): {{ latString }}, {{ lonString }}

OBS name: {{ obsName }}

OBS serial: {{ obsId }}

Deployment date: {{ deployDate }}

Deployment comments: {{ deployComments }}

Recovery date: {{ recoverDate }}

Recovery comments: {{ recoverComments }}

Length of deployment (days): {{ deploymentDays }}

Total clock drift (ms): {{ clockDrift }}

Average power consumption (mW): {{ meanPower }}

Remaining battery SOC: {{ batteryLevel }}%

## Introduction

{{ introText }}

## Seismic Data QC

{% for ch in seismic_channels %}
### {{ ch.channelName }}
SEED ID: {{ ch.seedID }}

Orientation: {{ azimuth }} / {{ dip }}

#### Full trace
![trace]({{ ch.traceLoc }})

#### Spectrogram
![spectrogram]({{ ch.specLoc }})

#### Power Spectral Density[^psd]
![PSD]({{ ch.psdLoc }})

[^psd]: Power spectral density curves are calculating using {{ windowLength }} windows, with {{ overlapPercent }}% overlap.

{% endfor %}

## Oceanographic Data

{% for ch in ocean_channels %}
![trace]({{ ch.traceLoc }})
{% endfor %}

## Battery Condition

{% for ch in power_channels %}
![trace]({{ ch.traceLoc }})
{% endfor %}

## Instrument State-of-Health

{% for ch in other_channels %}
![trace]({{ ch.traceLoc }})
{% endfor %}

