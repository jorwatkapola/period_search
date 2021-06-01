# generate data for a sine wave observed for 10 years with random noise added
no_observations=3650
no_days = 3650
x = np.linspace(0,no_days,num=no_observations)
period = 150
f=1/period
y = np.sin(x*f*np.pi*2)+np.random.normal(0,0.8,size=no_observations)

# randomly remove data
keep = np.random.choice([False,True], size=no_observations, p=[0.7,0.3])

# make observing epochs, removing data outside of yearly observins seasons
total_exposure_prop = 0.5
no_epochs = 10
epoch_len = no_days/no_epochs
epoch_len*total_exposure_prop

for n,i in enumerate(keep):
    if (n%epoch_len) > epoch_len*total_exposure_prop:
        keep[n] = 0
x = x[keep]
y = y[keep]

plt.scatter(x,y)
plt.show()
plt.xlabel("value")
plt.ylabel("amplitude")
plt.title("Sine wave with random noise, randomly missing data and regular epochs")

f = np.linspace(1/1000, 1/2, 10000)*np.pi*2
pgram = signal.lombscargle(x,y, f, normalize=True)
plt.axvline(x=period, c="magenta", zorder=1)
plt.plot(1/(f/(2.0*np.pi)), pgram, c="black", zorder=2)
plt.xlabel("period")
plt.ylabel("lomb scargle power")
plt.title("Magenta is the true period")

# for harm in range(2,7):
#     plt.axvline(x=period*harm, c="cyan", zorder=1)
#     plt.axvline(x=period/harm, c="cyan", zorder=1)


plt.show()

###############################################################

# generate data for a sine wave observed for 10 years with random noise added
data = np.loadtxt(data_path+"ogle_merged/sxp6.85.csv", delimiter=",", dtype=float)
x = data[:,0]
period = 150
period1 = 80
f=1/period
f1 = 1/period1
y = np.sin(x*f*np.pi*2)+np.random.normal(0,1,size=len(x))

y += np.sin(x*f1*np.pi*2)+np.random.normal(0,1,size=len(x))

plt.scatter(x,y)
plt.show()
plt.xlabel("value")
plt.ylabel("amplitude")
plt.title("Sine wave with random noise, randomly missing data and regular epochs")

f = np.linspace(1/1000, 1/2, 10000)*np.pi*2
pgram = signal.lombscargle(x,y, f, normalize=True)
plt.axvline(x=period, c="magenta", zorder=1)
plt.axvline(x=period1, c="magenta", zorder=1)
plt.plot(1/(f/(2.0*np.pi)), pgram, c="black", zorder=2)
plt.xlabel("period")
plt.ylabel("lomb scargle power")
plt.title("Magenta is the true period")

# for harm in range(2,7):
#     plt.axvline(x=period*harm, c="cyan", zorder=1)
#     plt.axvline(x=period/harm, c="cyan", zorder=1)


plt.show()