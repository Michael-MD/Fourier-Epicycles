import cv2
import matplotlib.pyplot as plt 
import numpy as np
import matplotlib.animation as animation
from matplotlib.patches import Circle
import sys
from tqdm import tqdm


plt.style.use('dark_background')


def prepare_image(img, N_desired=300):
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
	cycles = []
	coords_ordered = []
	p = coords[0]
	for _ in range(N:=len(coords)):
		coords_ordered.append(p)
		ind = np.argmin(
				d:=np.abs(p - coords)
			)
		print(d[ind])
		if d[ind] > 1.5:
			# jumped to different curve
			cycles.append(coords_ordered)
			coords_ordered = []

		p = coords[ind]
		coords[ind] = np.inf
	cycles.append(coords_ordered)

	if N_desired is None:
		N_desired = N

	skip_factor = np.max([int(N / N_desired),1])
	for i,cycle in enumerate(cycles):
		cycles[i] = cycle[0::skip_factor]

	for cycle in cycles:
		plt.plot(np.real(cycle), np.imag(cycle))
		plt.show()


	return cycles


coords = prepare_image(sys.argv[1], int(sys.argv[2]) if sys.argv[2] != 'None' else None )
# N = len(coords)

# # treat points as complex function and find contained frequencies
# f = np.fft.fftshift(np.fft.fftfreq(N))
# Xf = np.fft.fftshift(np.fft.fft(coords))

# x = lambda n: np.exp(1j*2*np.pi*f*n)*Xf/N
# xn = [x(n) for n in range(N)]
# x_sum = [np.sum(xni) for xni in xn]
# pn = [ np.insert(np.cumsum(xni), 0, 0+0j) for xni in xn]
# thetan, rn = np.angle(pn), np.abs(pn)
# rn_single = np.abs(xn)


# fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
# ax.axis('off')
# ax.set_ylim([np.pi*2,np.max(np.abs(coords))*1.1])

# p_bar = tqdm(range(frames:=N*2))

# class update_cls:
# 	def __init__(self):
# 		self.epicycles = []
# 		self.epicyle_radii = None

# 	def draw_circle(self,c,r):
# 		circle = Circle(c, r, transform=ax.transData._b, edgecolor='white', facecolor='none')
# 		s = ax.add_artist(circle)
# 		ax.set_aspect('equal')
# 		self.epicycles.append(s)

# 	def __call__(self,n):
# 		global s
# 		# update progress bar
# 		p_bar.n = n
# 		p_bar.refresh()

# 		# draw nth point
# 		if n < N:
# 			ax.plot(np.angle(x_sum[n-1:n+1]), np.abs(x_sum[n-1:n+1]), 'w')

# 		n %= N
# 		p = pn[n]

# 		# remove previous epicycle drawings
# 		if self.epicyle_radii is not None:
# 			[epicycle.remove() for epicycle in self.epicycles]
# 			[epicycle_radus.remove() for epicycle_radus in self.epicyle_radii]
# 			self.epicycles = []

# 		theta, r = thetan[n], rn[n]
# 		self.epicyle_radii = ax.plot(theta, r, 'w')

# 		# draw epicycle circles
# 		for ri, pi in zip(rn_single[n], p):
# 			self.draw_circle([np.real(pi),np.imag(pi)], ri)

# update = update_cls()

# anim = animation.FuncAnimation(fig, update, frames=frames, interval=1)
# if sys.argv[3] == '1':
# 	anim.save('animation.gif', writer='imagemagick', fps=60)

# plt.show()