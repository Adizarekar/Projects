#!/usr/bin/env python
# coding: utf-8

# In[1]:


import scipy.io
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.fft import fft
from scipy.signal import butter, lfilter, filtfilt,freqs
from scipy import stats
from tabulate import tabulate
BOr_W = scipy.io.loadmat('BOr_W.mat')
m1l_W = scipy.io.loadmat('M1l_W.mat')
m1r_W = scipy.io.loadmat('M1r_W.mat')
s1l_W = scipy.io.loadmat('s1l_W.mat')
s1r_W = scipy.io.loadmat('s1r_W.mat')
v2l_W = scipy.io.loadmat('v2l_W.mat')
v2r_W = scipy.io.loadmat('v2r_W.mat')


# In[2]:


eeg_signal = BOr_W['AW']
eeg_signal_m1l = m1l_W['AW']
eeg_signal_m1r = m1r_W['AW']
eeg_signal_s1l = s1l_W['AW']
eeg_signal_s1r = s1r_W['AW']
eeg_signal_v2l = v2l_W['AW']
eeg_signal_v2r = v2r_W['AW']


# In[3]:


fs = 1024  # Sampling rate in Hz
t = np.arange(0,len(eeg_signal) / fs, 1/fs)  

0# Define the frequency bands
delta_band = (0.5, 4)  # in radian
theta_band = (4, 8)
alpha_band = (8, 13)
beta_band = (13, 30)
gamma_band = (30, 50)

# Apply Butterworth bandpass filters
def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    y = lfilter(b, a, data,axis = 0)
    return y


# In[4]:


# Apply the filters to EEG signal BOr_W
delta_filteredBO = butter_bandpass_filter(eeg_signal, 2 * np.pi *delta_band[0], 2 * np.pi *delta_band[1], fs)
theta_filteredBO = butter_bandpass_filter(eeg_signal, 2 * np.pi *theta_band[0], 2 * np.pi *theta_band[1], fs)
alpha_filteredBO = butter_bandpass_filter(eeg_signal, 2 * np.pi *alpha_band[0], 2 * np.pi *alpha_band[1], fs)
beta_filteredBO = butter_bandpass_filter(eeg_signal, 2 * np.pi *beta_band[0], 2 * np.pi *beta_band[1], fs)
gamma_filteredBO = butter_bandpass_filter(eeg_signal, 2 * np.pi *gamma_band[0], 2 * np.pi *gamma_band[1], fs)


# In[5]:


# Apply the filters to EEG signal M1l_W
delta_filteredml = butter_bandpass_filter(eeg_signal_m1l, 2 * np.pi *delta_band[0], 2 * np.pi *delta_band[1], fs)
theta_filteredml = butter_bandpass_filter(eeg_signal_m1l, 2 * np.pi *theta_band[0], 2 * np.pi *theta_band[1], fs)
alpha_filteredml = butter_bandpass_filter(eeg_signal_m1l, 2 * np.pi *alpha_band[0], 2 * np.pi *alpha_band[1], fs)
beta_filteredml = butter_bandpass_filter(eeg_signal_m1l, 2 * np.pi *beta_band[0], 2 * np.pi *beta_band[1], fs)
gamma_filteredml = butter_bandpass_filter(eeg_signal_m1l, 2 * np.pi *gamma_band[0], 2 * np.pi *gamma_band[1], fs)


# In[6]:


# Apply the filters to EEG signal M1r_W
delta_filteredmr = butter_bandpass_filter(eeg_signal_m1r, 2 * np.pi *delta_band[0], 2 * np.pi *delta_band[1], fs)
theta_filteredmr = butter_bandpass_filter(eeg_signal_m1r, 2 * np.pi *theta_band[0], 2 * np.pi *theta_band[1], fs)
alpha_filteredmr = butter_bandpass_filter(eeg_signal_m1r, 2 * np.pi *alpha_band[0], 2 * np.pi *alpha_band[1], fs)
beta_filteredmr = butter_bandpass_filter(eeg_signal_m1r, 2 * np.pi *beta_band[0], 2 * np.pi *beta_band[1], fs)
gamma_filteredmr = butter_bandpass_filter(eeg_signal_m1r, 2 * np.pi *gamma_band[0], 2 * np.pi *gamma_band[1], fs)


# In[7]:


# Apply the filters to EEG signal S1l_W
delta_filteredsl = butter_bandpass_filter(eeg_signal_s1l, 2 * np.pi *delta_band[0], 2 * np.pi *delta_band[1], fs)
theta_filteredsl = butter_bandpass_filter(eeg_signal_s1l, 2 * np.pi *theta_band[0], 2 * np.pi *theta_band[1], fs)
alpha_filteredsl = butter_bandpass_filter(eeg_signal_s1l, 2 * np.pi *alpha_band[0], 2 * np.pi *alpha_band[1], fs)
beta_filteredsl = butter_bandpass_filter(eeg_signal_s1l, 2 * np.pi *beta_band[0], 2 * np.pi *beta_band[1], fs)
gamma_filteredsl = butter_bandpass_filter(eeg_signal_s1l, 2 * np.pi *gamma_band[0], 2 * np.pi *gamma_band[1], fs)


# In[8]:


# Apply the filters to EEG signal S1r_W
delta_filteredsr = butter_bandpass_filter(eeg_signal_s1r, 2 * np.pi *delta_band[0], 2 * np.pi *delta_band[1], fs)
theta_filteredsr = butter_bandpass_filter(eeg_signal_s1r, 2 * np.pi *theta_band[0], 2 * np.pi *theta_band[1], fs)
alpha_filteredsr = butter_bandpass_filter(eeg_signal_s1r, 2 * np.pi *alpha_band[0], 2 * np.pi *alpha_band[1], fs)
beta_filteredsr = butter_bandpass_filter(eeg_signal_s1r, 2 * np.pi *beta_band[0], 2 * np.pi *beta_band[1], fs)
gamma_filteredsr = butter_bandpass_filter(eeg_signal_s1r, 2 * np.pi *gamma_band[0], 2 * np.pi *gamma_band[1], fs)


# In[9]:


# Apply the filters to EEG signal V2l_W
delta_filteredvl = butter_bandpass_filter(eeg_signal_v2l, 2 * np.pi *delta_band[0], 2 * np.pi *delta_band[1], fs)
theta_filteredvl = butter_bandpass_filter(eeg_signal_v2l, 2 * np.pi *theta_band[0], 2 * np.pi *theta_band[1], fs)
alpha_filteredvl = butter_bandpass_filter(eeg_signal_v2l, 2 * np.pi *alpha_band[0], 2 * np.pi *alpha_band[1], fs)
beta_filteredvl = butter_bandpass_filter(eeg_signal_v2l, 2 * np.pi *beta_band[0], 2 * np.pi *beta_band[1], fs)
gamma_filteredvl = butter_bandpass_filter(eeg_signal_v2l, 2 * np.pi *gamma_band[0], 2 * np.pi *gamma_band[1], fs)


# In[10]:


# Apply the filters to EEG signal V2r_W
delta_filteredvr = butter_bandpass_filter(eeg_signal_v2r, 2 * np.pi *delta_band[0], 2 * np.pi *delta_band[1], fs)
theta_filteredvr = butter_bandpass_filter(eeg_signal_v2r, 2 * np.pi *theta_band[0], 2 * np.pi *theta_band[1], fs)
alpha_filteredvr = butter_bandpass_filter(eeg_signal_v2r, 2 * np.pi *alpha_band[0], 2 * np.pi *alpha_band[1], fs)
beta_filteredvr = butter_bandpass_filter(eeg_signal_v2r, 2 * np.pi *beta_band[0], 2 * np.pi *beta_band[1], fs)
gamma_filteredvr = butter_bandpass_filter(eeg_signal_v2r, 2 * np.pi *gamma_band[0], 2 * np.pi *gamma_band[1], fs)


# In[30]:


def plot_probability_histogram(data, title, bins=50):
    # Create a histogram without plotting it
    data = data.flatten()
    hist, edges = np.histogram(data, bins=bins, density=True)
    f_title = 'Probability Histogram of ' + title


    # Calculate bin widths
    bin_widths = edges[1:] - edges[:-1]

    # Calculate the probability density for each bin
    pdf = hist * bin_widths
    
    # Plot the PDF
    plt.bar(edges[:-1], pdf, width=bin_widths, edgecolor='black')

    # Add labels and title
    plt.xlabel('Value')
    plt.ylabel('Probability Density')
    plt.title(f_title)

    # Show the histogram
    plt.show()

plt.subplot(6, 1, 1)
plot_probability_histogram(eeg_signal[:4000], 'Raw Signal BO_W')
plt.subplot(6, 1, 2)
plot_probability_histogram(delta_filteredBO, 'Delta signal BO_W')
plt.subplot(6, 1, 3)
plot_probability_histogram(theta_filteredBO, 'Theta signal BO_W')
plt.subplot(6, 1, 4)
plot_probability_histogram(alpha_filteredBO, 'Alpha signal BO_W')
plt.subplot(6, 1, 5)
plot_probability_histogram(beta_filteredBO, 'Beta signal BO_W')
plt.subplot(6, 1, 6)
plot_probability_histogram(gamma_filteredBO, 'Gamma signal BO_W')


# In[12]:


plt.subplot(6, 1, 1)
plot_probability_histogram(eeg_signal_m1l, 'Raw Signal M1l_W')
plt.subplot(6, 1, 2)
plot_probability_histogram(delta_filteredml, 'Delta signal M1l_W')
plt.subplot(6, 1, 3)
plot_probability_histogram(theta_filteredml, 'Theta signal M1l_W')
plt.subplot(6, 1, 4)
plot_probability_histogram(alpha_filteredml, 'Alpha signal M1l_W')
plt.subplot(6, 1, 5)
plot_probability_histogram(beta_filteredml, 'Beta signal M1l_W')
plt.subplot(6, 1, 6)
plot_probability_histogram(gamma_filteredml, 'Gamma signal M1l_W')


# In[13]:


plt.subplot(6, 1, 1)
plot_probability_histogram(eeg_signal_m1r, 'Raw Signal M1r_W')
plt.subplot(6, 1, 2)
plot_probability_histogram(delta_filteredmr, 'Delta signal M1r_W')
plt.subplot(6, 1, 3)
plot_probability_histogram(theta_filteredmr, 'Theta signal M1r_W')
plt.subplot(6, 1, 4)
plot_probability_histogram(alpha_filteredmr, 'Alpha signal M1r_W')
plt.subplot(6, 1, 5)
plot_probability_histogram(beta_filteredmr, 'Beta signal M1r_W')
plt.subplot(6, 1, 6)
plot_probability_histogram(gamma_filteredmr, 'Gamma signal M1r_W')


# In[14]:


plt.subplot(6, 1, 1)
plot_probability_histogram(eeg_signal_s1l, 'Raw Signal S1l_W')
plt.subplot(6, 1, 2)
plot_probability_histogram(delta_filteredsl, 'Delta signal S1l_W')
plt.subplot(6, 1, 3)
plot_probability_histogram(theta_filteredsl, 'Theta signal S1l_W')
plt.subplot(6, 1, 4)
plot_probability_histogram(alpha_filteredsl, 'Alpha signal S1l_W')
plt.subplot(6, 1, 5)
plot_probability_histogram(beta_filteredsl, 'Beta signal S1l_W')
plt.subplot(6, 1, 6)
plot_probability_histogram(gamma_filteredsl, 'Gamma signal S1l_W')


# In[15]:


plt.subplot(6, 1, 1)
plot_probability_histogram(eeg_signal_s1r, 'Raw Signal S1r_W')
plt.subplot(6, 1, 2)
plot_probability_histogram(delta_filteredsr, 'Delta signal S1r_W')
plt.subplot(6, 1, 3)
plot_probability_histogram(theta_filteredsr, 'Theta signal S1r_W')
plt.subplot(6, 1, 4)
plot_probability_histogram(alpha_filteredsr, 'Alpha signal S1r_W')
plt.subplot(6, 1, 5)
plot_probability_histogram(beta_filteredsr, 'Beta signal S1r_W')
plt.subplot(6, 1, 6)
plot_probability_histogram(gamma_filteredsr, 'Gamma signal S1r_W')


# In[16]:


plt.subplot(6, 1, 1)
plot_probability_histogram(eeg_signal_v2l, 'Raw Signal V2l_W')
plt.subplot(6, 1, 2)
plot_probability_histogram(delta_filteredvl, 'Delta signal v2l_W')
plt.subplot(6, 1, 3)
plot_probability_histogram(theta_filteredvl, 'Theta signal v2l_W')
plt.subplot(6, 1, 4)
plot_probability_histogram(alpha_filteredvl, 'Alpha signal v2l_W')
plt.subplot(6, 1, 5)
plot_probability_histogram(beta_filteredvl, 'Beta signal v2l_W')
plt.subplot(6, 1, 6)
plot_probability_histogram(gamma_filteredvl, 'Gamma signal v2l_W')


# In[17]:


plt.subplot(6, 1, 1)
plot_probability_histogram(eeg_signal_v2r, 'Raw Signal V2r_W')
plt.subplot(6, 1, 2)
plot_probability_histogram(delta_filteredvr, 'Delta signal v2r_W')
plt.subplot(6, 1, 3)
plot_probability_histogram(theta_filteredvr, 'Theta signal v2r_W')
plt.subplot(6, 1, 4)
plot_probability_histogram(alpha_filteredvr, 'Alpha signal v2r_W')
plt.subplot(6, 1, 5)
plot_probability_histogram(beta_filteredvr, 'Beta signal v2r_W')
plt.subplot(6, 1, 6)
plot_probability_histogram(gamma_filteredvr, 'Gamma signal v2r_W')


# In[28]:


from scipy.stats import anderson

# Perform Anderson-Darling test
result = anderson(eeg_signal[:2000].flatten())

# Print the test statistic and critical values
print(f"Test Statistic: {result.statistic}")
print("Critical Values:")
for i, crit_val in enumerate(result.critical_values):
    print(f"Level {result.significance_level[i]}%: {crit_val}")

# Check if the test statistic is less than the critical value at a chosen significance level (e.g., 5%)
if result.statistic < result.critical_values[2]:
    print("Fail to reject the null hypothesis. Data may be normally distributed.")
else:
    print("Reject the null hypothesis. Data may not be normally distributed.")


# In[19]:


import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Example data (replace this with your actual data)
data = np.random.normal(loc=0, scale=1, size=5000)

# Create a Q-Q plot
sm.qqplot(delta_filteredBO, line='s', fit=True)
plt.title("Q-Q Plot")
plt.show()


# In[20]:


v_test_BO = [["", "Raw Signal BO", "Delta BO", "Theta BO", "Alpha BO", "Beta BO", "Gamma BO"],             ["Statistic", stats.shapiro(eeg_signal).statistic, stats.shapiro(delta_filteredBO).statistic,              stats.shapiro(theta_filteredBO).statistic, stats.shapiro(alpha_filteredBO).statistic,              stats.shapiro(beta_filteredBO).statistic, stats.shapiro(gamma_filteredBO).statistic],             ["P-Value", stats.shapiro(eeg_signal).pvalue, stats.shapiro(delta_filteredBO).pvalue,              stats.shapiro(theta_filteredBO).pvalue, stats.shapiro(alpha_filteredBO).pvalue,              stats.shapiro(beta_filteredBO).pvalue, stats.shapiro(gamma_filteredBO).pvalue]]
print(tabulate(v_test_BO, headers='firstrow'))


# In[21]:


v_test_Ml = [["", "Raw Signal Ml", "Delta Ml", "Theta Ml", "Alpha Ml", "Beta Ml", "Gamma Ml"],             ["Statistic", stats.shapiro(eeg_signal_m1l).statistic,stats.shapiro(delta_filteredml).statistic,              stats.shapiro(theta_filteredml).statistic, stats.shapiro(alpha_filteredml).statistic,              stats.shapiro(beta_filteredml).statistic,stats.shapiro(gamma_filteredml).statistic],            ["P-Value", stats.shapiro(eeg_signal_m1l).pvalue,stats.shapiro(delta_filteredml).pvalue,              stats.shapiro(theta_filteredml).pvalue, stats.shapiro(alpha_filteredml).pvalue,              stats.shapiro(beta_filteredml).pvalue,stats.shapiro(gamma_filteredml).pvalue]]
print(tabulate(v_test_Ml, headers='firstrow'))


# In[22]:


v_test_Mr = [["", "Raw Signal Mr", "Delta Mr", "Theta Mr", "Alpha Mr", "Beta Mr", "Gamma Mr"],             ["Statistic", stats.shapiro(eeg_signal_m1r).statistic,stats.shapiro(delta_filteredmr).statistic,              stats.shapiro(theta_filteredmr).statistic, stats.shapiro(alpha_filteredmr).statistic,              stats.shapiro(beta_filteredmr).statistic,stats.shapiro(gamma_filteredmr).statistic],            ["P-Value", stats.shapiro(eeg_signal_m1r).pvalue,stats.shapiro(delta_filteredmr).pvalue,              stats.shapiro(theta_filteredmr).pvalue, stats.shapiro(alpha_filteredmr).pvalue,              stats.shapiro(beta_filteredmr).pvalue,stats.shapiro(gamma_filteredmr).pvalue]]
print(tabulate(v_test_Mr, headers='firstrow'))


# In[23]:


v_test_Sl = [["", "Raw Signal Sl", "Delta Sl", "Theta Sl", "Alpha Sl", "Beta Sl", "Gamma Sl"],             ["Statistics", stats.shapiro(eeg_signal_s1l).statistic,stats.shapiro(delta_filteredsl).statistic,              stats.shapiro(theta_filteredsl).statistic, stats.shapiro(alpha_filteredsl).statistic,              stats.shapiro(beta_filteredsl).statistic,stats.shapiro(gamma_filteredsl).statistic],             ["P-Value", stats.shapiro(eeg_signal_s1l).pvalue,stats.shapiro(delta_filteredsl).pvalue,              stats.shapiro(theta_filteredsl).pvalue, stats.shapiro(alpha_filteredsl).pvalue,              stats.shapiro(beta_filteredsl).pvalue,stats.shapiro(gamma_filteredsl).pvalue]]
print(tabulate(v_test_Sl, headers='firstrow'))


# In[24]:


v_test_Sr = [["", "Raw Signal Sr", "Delta Sr", "Theta Sr", "Alpha Sr", "Beta Sr", "Gamma Sr"],             ["Statistics", stats.shapiro(eeg_signal_s1r).statistic,stats.shapiro(delta_filteredsr).statistic,              stats.shapiro(theta_filteredsr).statistic, stats.shapiro(alpha_filteredsr).statistic,              stats.shapiro(beta_filteredsr).statistic,stats.shapiro(gamma_filteredsr).statistic],             ["P-Value", stats.shapiro(eeg_signal_s1r).pvalue,stats.shapiro(delta_filteredsr).pvalue,              stats.shapiro(theta_filteredsr).pvalue, stats.shapiro(alpha_filteredsr).pvalue,              stats.shapiro(beta_filteredsr).pvalue,stats.shapiro(gamma_filteredsr).pvalue]]
print(tabulate(v_test_Sr, headers='firstrow'))


# In[25]:


v_test_Vl = [["", "Raw Signal Vl", "Delta Vl", "Theta Vl", "Alpha Vl", "Beta Vl", "Gamma Vl"],             ["Statistics", stats.shapiro(eeg_signal_v2l).statistic,stats.shapiro(delta_filteredvl).statistic,              stats.shapiro(theta_filteredvl).statistic, stats.shapiro(alpha_filteredvl).statistic,              stats.shapiro(beta_filteredvl).statistic,stats.shapiro(gamma_filteredvl).statistic],             ["P-Value", stats.shapiro(eeg_signal_v2l).pvalue,stats.shapiro(delta_filteredvl).pvalue,              stats.shapiro(theta_filteredvl).pvalue, stats.shapiro(alpha_filteredvl).pvalue,              stats.shapiro(beta_filteredvl).pvalue,stats.shapiro(gamma_filteredvl).pvalue]]
print(tabulate(v_test_Vl, headers='firstrow'))


# In[26]:


v_test_Vr = [["", "Raw Signal Vr", "Delta Vr", "Theta Vr", "Alpha Vr", "Beta Vr", "Gamma Vr"],             ["Statistics", stats.shapiro(eeg_signal_v2r).statistic,stats.shapiro(delta_filteredvr).statistic,              stats.shapiro(theta_filteredvr).statistic, stats.shapiro(alpha_filteredvr).statistic,              stats.shapiro(beta_filteredvr).statistic,stats.shapiro(gamma_filteredvr).statistic],             ["P-Value", stats.shapiro(eeg_signal_v2r).pvalue,stats.shapiro(delta_filteredvr).pvalue,              stats.shapiro(theta_filteredvr).pvalue, stats.shapiro(alpha_filteredvr).pvalue,              stats.shapiro(beta_filteredvr).pvalue,stats.shapiro(gamma_filteredvr).pvalue]]
print(tabulate(v_test_Vr, headers='firstrow'))


# In[ ]:




