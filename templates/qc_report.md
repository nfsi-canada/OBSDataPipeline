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

{% if meanPressure or meanTemperature %}
Average seafloor conditions: Pressure {% if meanPressure %}{{ meanPressure }} Pa{% else %}n/a{% endif %}, Temperature {% if meanTemperature %}{{ meanTemperature }} &deg;C{% else %}n/a{% endif %} 

{% endif %}
{% if tiltAtDeploy or tiltAtRecovery %}
Tilt estimates in OBS's internal reference frame (angle measured from Earth vertical): 

{% if tiltAtDeploy %}
- At deployment: Angle {{ tiltAtDeploy.angle }}&deg;, Direction {{ tiltAtDeploy.azimuth }}&deg;
{% endif %}
{% if tiltAtRecovery %}
- At recovery: Angle {{ tiltAtRecovery.angle }}&deg;, Direction {{ tiltAtRecovery.azimuth }}&deg;
{% endif %}
{% if tiltRotation %}
- Apparent tilt rotation over deployment period: {{ tiltRotation }}&deg;

{% endif %}
{% endif %}

\newpage{}

# Introduction

{% if intro_pt1 %}
{{ intro_pt1 }}
{% endif %}

{% if introText %}
{{ introText }}
{% endif %}

# General QC

This report analyzes data recorded while the instrument is physically at the seabed. Touchdown and release times are determined by manual inspection of the auxiliary data channels where possible. Throughout this report, power spectral density curves are calculated using {{ psdWindowLength }} Hann windows with {{ psdOverlapPercent }}% overlap, following an average periodogram method similar to that described by McNamara & Buland (2004).

{% if seismic_ignored %}
Seismoacoustic data (seismometer and/or hydrophone) may or may not exist in the data package analyzed. Such data channels have been ignored in preparing this report.

{% elif seismic_limited %}
Analysis of seismoacoustic data channels (seismometer and/or hydrophone), if present in this data package, is limited to assessment of data extent and readability. No other analysis or plots have been generated for such channels in preparing this report.

{% endif %}

{% if subzero %}
Recorded values on the external temperature sensor indicate readings likely dropped below 0&deg;C during this deployment. The sensor does not read accurately for this range, causing values to wrap to the top of the available data range and display suspected non-physical effects. Processing has been done for this report to attempt to recover sub-zero temperature readings. Such readings should be treated as approximate only.

{% elif temp_wrap %}
Recorded values on the external temperature sensor indicate readings likely dropped below 0&deg;C during this deployment. The sensor does not read accurately for this range, causing values to wrap to the top of the available data range and display suspected non-physical effects. This data should not be used for detailed analysis without careful weeding.

{% endif %}

{% if channelList %}
Recorded data channels analyzed (time at seafloor):

| Channel | Start Time | End Time | Sampling Rate (Hz) |
|:--:|:---:|:---:|:--:|
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

{% if seismic_channels %}
# Seismic Data

{% for ch in seismic_channels %}
## {{ ch.channelName }}
SEED ID: {{ ch.seedID }}

{% if seismic_limited %}
Data details:

----------------  ----------------------
Start timestamp    {{ ch.start_string }}
End timestamp        {{ ch.end_string }}
Sampling rate       {{ ch.sampling }} Hz
----------------  ----------------------

{% if ch.gaps %}
The following data gaps are present in the recorded data:

| Start Time | End Time | Length (s) | Samples |
|:---|:---|-:|-:|
{% for gap in ch.gaps %}
| {{ gap.start }} | {{ gap.end }} | {{ gap.sec }} | {{ gap.samp }} |
{% endfor %}

{% else %}
No gaps are present in the recorded data.

{% endif %}

{% else %}
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
\newpage{}

### Power Spectral Density

PSD curves are binned by frequency and amplitude to generate density heatmaps (probabilistic power spectral density).{% if not ch.hydrophone %} Black curves overlain on these plots are the Peterson high and low global noise models (NHNM and NLNM; Peterson, 1993).{% endif %} 

{% for psd in ch.psdLoc %}
{% if ch.hydrophone %}
![PPSD plot for channel {{ ch.seedID }} for {{ psd.start }} to {{ psd.end }}]({{ psd.image }})
{% else %}
![Acceleration PPSD plot for channel {{ ch.seedID }} for {{ psd.start }} to {{ psd.end }}]({{ psd.image }})
{% endif %}

{% endfor %}
{% endif %}

\newpage{}
{% endif %}

{% endfor %}
{% if seismic_limited %}
\newpage{}
{% endif %}
{% endif %}

# Oceanographic Data

{% for ch in ocean_channels %}
## {{ ch.channelName }}
SEED ID: {{ ch.seedID }}

![Recorded data for channel {{ ch.seedID }}]({{ ch.traceLoc }})

{% if ch.despikedPlot %}
This channel may show periodic spikes due to the data write cycle of the Aquarius, which are not representative of real environmental conditions.

![Despiked data for channel {{ ch.seedID }}]({{ ch.despikedPlot }})

{% endif %}
{% if ch.rollPlot %}
![Average reading for channel {{ ch.seedID }} for a {{ ch.window_str }} rolling window with 66% overlap]({{ ch.rollPlot }})

{% endif %}
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
