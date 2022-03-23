# OBS Data Quality Report - {{ obsName }}

## Summary

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

## Introduction

{{ introText }}

## Seismic Data QC

{% for ch in seismic_channels %}
### {{ ch.channelName }}
SEED ID: {{ ch.seedID }}

{% if ch.azimuth %}
Orientation: {{ ch.azimuth }} / {{ ch.dip }}
{% endif %}

#### Full trace
![Recorded data for channel {{ ch.seedID }}]({{ ch.traceLoc }})

#### Spectrogram
![Spectrogram of channel {{ ch.seedID }}]({{ ch.specLoc }})

#### Power Spectral Density[^psd]
![Power spectral density curves for channel {{ ch.seedID }}]({{ ch.psdLoc }})

{% endfor %}

## Oceanographic Data

{% for ch in ocean_channels %}
### {{ ch.channelName }}
SEED ID: {{ ch.seedID }}

![Recorded data for channel {{ ch.seedID }}]({{ ch.traceLoc }})

{% endfor %}

## Battery Condition

{% for ch in power_channels %}
### {{ ch.channelName }}
SEED ID: {{ ch.seedID }}

![Recorded data for channel {{ ch.seedID }}]({{ ch.traceLoc }})

{% endfor %}

## Instrument State-of-Health

{% for ch in health_channels %}
### {{ ch.channelName }}
SEED ID: {{ ch.seedID }}

![Recorded data for channel {{ ch.seedID }}]({{ ch.traceLoc }})

{% endfor %}

[^psd]: Power spectral density curves are calculating using {{ psdWindowLength }} windows, with {{ psdOverlapPercent }}% overlap.
