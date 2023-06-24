import cv2
import matplotlib.pyplot as plt 
import numpy as np
import matplotlib.animation as animation
from matplotlib.patches import Circle
import sys
from tqdm import tqdm


plt.style.use('dark_background')


def prepare_image(img, N_desired=100):
	img_gray = np.average(
			cv2.imread(img)
		,axis=-1)

	# Blur the image for better edge detection
	img_blur = cv2.GaussianBlur(img_gray, (3,3), 0) 

	# Canny Edge Detection
	img_blur = np.uint8(img_blur)
	edges = cv2.Canny(image=img_blur, threshold1=100, threshold2=200) # Canny Edge Detection

	# extract edge points
	px, py = img_gray.shape
	gx, gy = np.linspace(0,px-1, px), np.linspace(0,py-1, py)
	gx, gy = np.meshgrid(gy, gx)
	edges_bool = edges == 255
	coords= (re:=gx[edges_bool]) + 1j * (im:=-gy[edges_bool])

	# shift image to centre
	coords -= np.average(re) + 1j * np.average(im)


	# arrange points based on total ordering since simple graph assumed 
	# doesn't include initial point to ensure periodic boundary conditions
	coords_ordered = []
	p = coords[0]
	for _ in range(N:=len(coords)):
		coords_ordered.append(p)
		ind = np.argmin(
				np.abs(p - coords)
			)
		p = coords[ind]
		coords[ind] = np.inf

	if N_desired is None:
		N_desired = N
	print(N)
	skip_factor = np.max([int(N / N_desired),1])
	return coords_ordered[0::skip_factor]


coords = prepare_image(sys.argv[1], int(sys.argv[2]))
N = len(coords)

# treat points as complex function and find contained frequencies
f = np.fft.fftshift(np.fft.fftfreq(N))
Xf = np.fft.fftshift(np.fft.fft(coords))

xn = lambda n: np.exp(1j*2*np.pi*f*n)*Xf/N
x = [xn(n) for n in range(N)]
x_sum = [np.sum(xi) for xi in x]

fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
ax.axis('off')

p_bar = tqdm(range(frames:=N*2))

def draw_circle(c,r):
	circle = Circle(c, r, transform=ax.transData._b, edgecolor='white', facecolor='none')
	ax.add_artist(circle)
	ax.set_aspect('equal')

def update(n):
	ax.clear()
	ax.axis('off')
	ax.set_ylim([np.pi*2,np.max(np.abs(coords))*1.5])

	p_bar.n = n
	p_bar.refresh()

	m = np.min([n, N-1])
	n %= N

	p = np.cumsum(x[n])
	p = np.insert(p, 0, 0+0j)
	theta, r = np.angle(p), np.abs(p)
	
	for ri, pi in zip(np.abs(x[n]), p):
		draw_circle([np.real(pi),np.imag(pi)], ri)
	
	ax.plot(theta, r)
	ax.plot(np.angle(x_sum[:m+1]), np.abs(x_sum[:m+1]))


a = animation.FuncAnimation(fig, update, frames=frames, interval=1)
# a.save('animation.gif', writer='imagemagick', fps=60)

plt.show()